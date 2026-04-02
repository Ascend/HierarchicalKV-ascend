/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file find_utils.h
 * @brief Device-side find helpers (digest probe). Included after `types.h`
 *        materializes `Bucket` / `OccupyResult`; depends on `ldg_l2nc_l1c` from
 *        `utils.h` via `types.h` include chain — do not include from `utils.h`.
 */
#pragma once

#include "types.h"

namespace npu {
namespace hkv {

template <class K, class V, class S>
__forceinline__ __device__ OccupyResult find_without_lock(
    __gm__ Bucket<K, V, S>* __restrict__ bucket,
    const K desired_key,
    uint32_t key_pos,
    const VecD_Comp target_digests,
    uint32_t& target_pos,
    const uint32_t bucket_capacity) {
  using BUCKET = Bucket<K, V, S>;
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  VecD_Comp empty_digests_val = empty_digests<K>();
  for (uint32_t offset = 0; offset < bucket_capacity + STRIDE;
       offset += STRIDE) {
    uint32_t pos_cur = align_to<STRIDE>(key_pos);
    pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

    __gm__ D* digests_ptr =
        BUCKET::digests(bucket->keys_, bucket_capacity, pos_cur);
    VecD_Comp probe_digests =
        ldg_l2nc_l1c(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));
    uint32_t possible_pos = 0;
    uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
    cmp_result &= 0x01010101;
    do {
      if (cmp_result == 0) {
        break;
      }
      uint32_t index =
          (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      possible_pos = pos_cur + index;
      const K current_key = ldg_l2nc_l1c(bucket->keys_ + possible_pos);
      if (current_key == desired_key) {
        target_pos = possible_pos;
        return OccupyResult::DUPLICATE;
      }
    } while (true);
    cmp_result = vcmpeq4(probe_digests, empty_digests_val);
    cmp_result &= 0x01010101;
    do {
      if (cmp_result == 0) {
        break;
      }
      uint32_t index =
          (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      possible_pos = pos_cur + index;
      if (offset == 0 && possible_pos < key_pos) {
        continue;
      }
      const K current_key = ldg_l2nc_l1c(bucket->keys_ + possible_pos);
      if (current_key == static_cast<K>(EMPTY_KEY)) {
        return OccupyResult::OCCUPIED_EMPTY;
      }
    } while (true);
  }
  return OccupyResult::CONTINUE;
}

/*
 * find_and_lock: TILE_SIZE threads cooperate to find/lock a position in bucket. Only support TILE_SIZE = 32.
 * Combines find-key, find-empty, and eviction into one unified interface.
 * After return, key_pos holds the locked position and evicted_key holds the
 * original key at that position (meaningful for EVICT).
 */
template <typename K, typename S, int32_t TILE_SIZE = 32>
__forceinline__ __device__ OccupyResult find_and_lock(
    __gm__ K* bucket_keys, __gm__ S* bucket_scores,
    const uint32_t bucket_max_size, const K& key, const S& score,
    uint32_t& key_pos, K& evicted_key, const uint32_t lane_id) {

  const uint32_t start_pos = key_pos;

  // Phase 1: Find existing key or empty slot
  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    uint32_t current_pos =
        (start_pos + tile_offset + lane_id) % bucket_max_size;

    // Step 1: 每个线程同时对自己的位置 CAS(key → LOCKED_KEY)
    K expected_key;
    uint32_t vote;
    do {
      expected_key = asc_atomic_cas(
          bucket_keys + current_pos, key, static_cast<K>(LOCKED_KEY));
      bool locked = (expected_key == key);

      vote = asc_ballot(locked);
      if (vote) {
        int32_t src_lane = __ffs(static_cast<int32_t>(vote)) - 1;
        key_pos = asc_shfl(current_pos, src_lane, TILE_SIZE);
        return OccupyResult::DUPLICATE;
      }

      vote = asc_ballot(
          expected_key == static_cast<K>(LOCKED_KEY));
      if (vote) {
        continue;
      }
      vote = asc_ballot(
          expected_key == static_cast<K>(EMPTY_KEY));
      if (vote) {
        break;
      }
    } while (vote != 0);

    // Step 2: 逐个尝试抢占空位
    while (vote) {
      int32_t src_lane = __ffs(static_cast<int32_t>(vote)) - 1;
      K cas_expected = static_cast<K>(EMPTY_KEY);
      if (static_cast<int32_t>(lane_id) == src_lane) {
        cas_expected = asc_atomic_cas(
            bucket_keys + current_pos,
            static_cast<K>(EMPTY_KEY),
            static_cast<K>(LOCKED_KEY));
      }
      cas_expected = asc_shfl(cas_expected, src_lane, TILE_SIZE);
      if (cas_expected == static_cast<K>(EMPTY_KEY)) {
        key_pos = asc_shfl(current_pos, src_lane, TILE_SIZE);
        return OccupyResult::OCCUPIED_EMPTY;
      }
      if (cas_expected == key ||
          cas_expected == static_cast<K>(LOCKED_KEY)) {
        return OccupyResult::CONTINUE;
      }
      vote -= (static_cast<uint32_t>(1) << src_lane);
    }
  }

  // Phase 2: Eviction (bucket full, key not found)
  S local_min_score = static_cast<S>(MAX_SCORE);
  uint32_t local_min_pos = 0;
  K local_min_key = static_cast<K>(EMPTY_KEY);

  // 2.1 遍历桶找最低分，先读 score，仅在更小时才自旋读 key
  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    uint32_t current_pos =
        (start_pos + tile_offset + lane_id) % bucket_max_size;

    S temp_score =
        __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(bucket_scores + current_pos);
    if (temp_score < local_min_score) {
      K current_key;
      while ((current_key =
                  __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                        L1CacheType::NON_CACHEABLE>(
                      bucket_keys + current_pos)) ==
             static_cast<K>(LOCKED_KEY)) {
        asc_threadfence();
      }
      if (current_key != static_cast<K>(EMPTY_KEY)) {
        local_min_key = current_key;
        local_min_score = temp_score;
        local_min_pos = current_pos;
      }
    }
  }

  // 2.2 归约获得全局最小 score
  S global_min_score = local_min_score;
  for (int32_t off = TILE_SIZE / 2; off > 0; off /= 2) {
    S other_score = asc_shfl_xor(global_min_score, off, TILE_SIZE);
    if (other_score < global_min_score) {
      global_min_score = other_score;
    }
  }

  // 2.3 分数不足，无法准入
  if (score < global_min_score) {
    return OccupyResult::REFUSED;
  }

  // 2.4 找到持有最低分的 lane，由该 lane 尝试 CAS 抢占
  uint32_t vote = asc_ballot(local_min_score <= global_min_score);
  if (vote) {
    int32_t src_lane = __ffs(static_cast<int32_t>(vote)) - 1;
    bool result = false;
    if (static_cast<int32_t>(lane_id) == src_lane) {
      evicted_key = local_min_key;
      auto try_key = asc_atomic_cas(bucket_keys + local_min_pos,
                                     local_min_key,
                                     static_cast<K>(LOCKED_KEY));
      if (try_key == local_min_key) {
        if (__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(
                bucket_scores + local_min_pos) <= global_min_score) {
          key_pos = local_min_pos;
          result = true;
        } else {
          (void)asc_atomic_exch(
              bucket_keys + local_min_pos, local_min_key);
        }
      }
    }
    result = static_cast<bool>(
        asc_shfl(static_cast<int32_t>(result), src_lane, TILE_SIZE));
    if (result) {
      key_pos = asc_shfl(key_pos, src_lane, TILE_SIZE);
      evicted_key = asc_shfl(evicted_key, src_lane, TILE_SIZE);
      return (evicted_key == static_cast<K>(RECLAIM_KEY))
                 ? OccupyResult::OCCUPIED_RECLAIMED
                 : OccupyResult::EVICT;
    }
  }

  return OccupyResult::CONTINUE;
}
}  // namespace hkv
}  // namespace npu
