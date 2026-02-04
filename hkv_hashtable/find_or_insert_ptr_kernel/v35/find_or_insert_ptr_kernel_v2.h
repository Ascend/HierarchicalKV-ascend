/* *
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

/* !
 * \file find_or_insert_ptr_kernel_v2.h
 * \brief find_or_insert_ptr_kernel_v2
 */

#ifndef ASCENDC_FIND_OR_INSERT_PTR_KERNEL_V2_H_
#define ASCENDC_FIND_OR_INSERT_PTR_KERNEL_V2_H_

#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "../../../include/score_functor.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_or_insert_ptr_kernel_vf_v2(
    GM_ADDR buckets_gm, GM_ADDR buckets_size_gm, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys_gm,
    GM_ADDR value_ptrs_gm, GM_ADDR scores_gm, GM_ADDR key_ptrs_gm, uint64_t n,
    GM_ADDR founds_gm, uint64_t global_epoch, uint64_t cur_score,
    uint32_t blockIdx, uint64_t thread_all) {
  if (!buckets_gm) {
    return;
  }
  if (!buckets_size_gm) {
    return;
  }
  if (!keys_gm) {
    return;
  }
  if (!value_ptrs_gm) {
    return;
  }
  if (!founds_gm) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_gm);
  __gm__ V* __gm__* __restrict__ value_ptrs =
      reinterpret_cast<__gm__ V * __gm__*>(value_ptrs_gm);
  __gm__ K* __gm__* __restrict__ key_ptrs =
      reinterpret_cast<__gm__ K * __gm__*>(key_ptrs_gm);
  __gm__ bool* __restrict__ founds = reinterpret_cast<__gm__ bool*>(founds_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);

  constexpr uint32_t LOAD_LEN_S = sizeof(byte16) / sizeof(S);

  for (uint64_t kv_idx = blockIdx * blockDim.x + threadIdx.x; kv_idx < n;
       kv_idx += thread_all) {
    // 1. 每个线程处理一个key
    K key = keys[kv_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      founds[kv_idx] = false;
      value_ptrs[kv_idx] = nullptr;
      continue;
    }
    S score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);

    // 2. 计算key的hash值 && 定位key
    const K hashed_key = Murmur3HashDevice(key);
    VecD_Comp target_digests = digests_from_hashed<K>(hashed_key);
    uint64_t global_idx =
        static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
    uint32_t key_pos = global_idx % bucket_capacity;
    uint64_t bkt_idx = global_idx / bucket_capacity;

    __gm__ int32_t* bucket_size_ptr = buckets_size + bkt_idx;
    __gm__ BUCKET* bucket = buckets + bkt_idx;
    uint32_t bucket_size = *bucket_size_ptr;
    __gm__ K* bucket_keys_ptr = bucket->keys_;
    __gm__ V* bucket_values_ptr = bucket->vectors;

    // 3. 遍历桶，找key
    OccupyResult occupy_result{OccupyResult::INITIAL};
    // Load `STRIDE` digests every time.
    constexpr uint32_t STRIDE = sizeof(VecD_Load) / sizeof(D);
    // One more loop to handle empty keys.
    for (int offset = 0; offset < bucket_capacity + STRIDE; offset += STRIDE) {
      if (occupy_result != OccupyResult::INITIAL) break;

      uint32_t pos_cur = align_to<STRIDE>(key_pos);
      pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

      __gm__ D* digests_ptr =
          BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
      VecD_Load digests_vec =
          *(reinterpret_cast<__gm__ VecD_Load*>(digests_ptr));
      VecD_Comp digests_arr[4] = {digests_vec.x, digests_vec.y, digests_vec.z,
                                  digests_vec.w};

      for (int i = 0; i < 4; i++) {
        VecD_Comp probe_digests = digests_arr[i];
        uint32_t possible_pos = 0;
        bool result = false;
        // Perform a vectorized comparison by byte,
        // and if they are equal, set the corresponding byte in the result to
        // 0xff.
        int cmp_result = vcmpeq4(probe_digests, target_digests);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) break;
          // NPU uses little endian,
          // and the lowest byte in register stores in the lowest address.
          uint32_t index = (__ffs(cmp_result) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + i * 4 + index;
          auto current_key_ptr = BUCKET::keys(bucket_keys_ptr, possible_pos);
          K expected_key = key;
          // Modifications to the bucket will not happen before this instruction.
          result = (AscendC::Simt::AtomicCas(current_key_ptr, expected_key,
                                             static_cast<K>(LOCKED_KEY)) ==
                    expected_key);
        } while (!result);
        if (result) {
          occupy_result = OccupyResult::DUPLICATE;
          key_pos = possible_pos;
          ScoreFunctor::update_score_only(bucket_keys_ptr, key_pos, scores,
                                          kv_idx, score, bucket_capacity,
                                          false);
          break;
        } else if (bucket_size == bucket_capacity) {
          continue;
        }
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) break;
          uint32_t index = (__ffs(cmp_result) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + i * 4 + index;
          if (offset == 0 && possible_pos < key_pos) continue;
          auto current_key_ptr = BUCKET::keys(bucket_keys_ptr, possible_pos);
          K expected_key = static_cast<K>(EMPTY_KEY);
          result = (AscendC::Simt::AtomicCas(current_key_ptr, expected_key,
                                             static_cast<K>(LOCKED_KEY)) ==
                    expected_key);
        } while (!result);
        if (result) {
          occupy_result = OccupyResult::OCCUPIED_EMPTY;
          key_pos = possible_pos;
          ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                           kv_idx, score, bucket_capacity,
                                           get_digest<K>(key), true);

          atomicAdd(bucket_size_ptr, 1);
          break;
        }
      }
    }

    while (occupy_result == OccupyResult::INITIAL) {
      __gm__ S* bucket_scores_ptr =
          BUCKET::scores(bucket_keys_ptr, bucket_capacity, 0);
      S min_score = MAX_SCORE;
      int min_pos = -1;
      for (int i = 0; i < bucket_capacity; i += LOAD_LEN_S) {
        S temp_scores[LOAD_LEN_S];
        *reinterpret_cast<byte16*>(temp_scores) =
            *reinterpret_cast<__gm__ byte16*>(bucket_scores_ptr + i);
#pragma unroll
        for (int j = 0; j < LOAD_LEN_S; j += 1) {
          S temp_score = temp_scores[j];
          if (temp_score < min_score) {
            auto verify_key_ptr = BUCKET::keys(bucket_keys_ptr, i + j);
            auto verify_key = *verify_key_ptr;
            if (verify_key != static_cast<K>(LOCKED_KEY) &&
                verify_key != static_cast<K>(EMPTY_KEY)) {
              min_score = temp_score;
              min_pos = i + j;
            }
          }
        }
      }

      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);
      if (score <= min_score) {
        occupy_result = OccupyResult::REFUSED;
        break;
      }
      auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos);
      auto expected_key = *min_score_key;
      if (expected_key != static_cast<K>(LOCKED_KEY) &&
          expected_key != static_cast<K>(EMPTY_KEY)) {
        bool result = (AscendC::Simt::AtomicCas(min_score_key, expected_key,
                                                static_cast<K>(LOCKED_KEY)) ==
                       expected_key);
        if (result) {
          __gm__ S* min_score_ptr =
              BUCKET::scores(bucket_keys_ptr, bucket_capacity, min_pos);
          auto verify_score = *min_score_ptr;
          if (verify_score <= min_score) {
            key_pos = min_pos;
            ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                             kv_idx, score, bucket_capacity,
                                             get_digest<K>(key), true);

            if (expected_key == static_cast<K>(RECLAIM_KEY)) {
              occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
              atomicAdd(bucket_size_ptr, 1);
            } else {
              occupy_result = OccupyResult::EVICT;
            }
          } else {
            *min_score_key = expected_key;
          }
        }
      }
    }

    if (occupy_result == OccupyResult::REFUSED) {
      value_ptrs[kv_idx] = nullptr;
    } else {
      value_ptrs[kv_idx] = bucket_values_ptr + key_pos * dim;
      auto key_address = BUCKET::keys(bucket_keys_ptr, key_pos);
      *key_address = key;
    }
    founds[kv_idx] = occupy_result == OccupyResult::DUPLICATE;
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_OR_INSERT_PTR_KERNEL_V2_H_
