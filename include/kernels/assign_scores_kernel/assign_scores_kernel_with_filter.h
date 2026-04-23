/* *
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

/* !
 * \file assign_scores_kernel_with_filter.h
 * \brief assign_scores_kernel_with_filter
 */

#ifndef ASCENDC_ASSIGN_SCORES_KERNEL_WITH_FILTER_H_
#define ASCENDC_ASSIGN_SCORES_KERNEL_WITH_FILTER_H_

#include <kernel_operator.h>
#include <simt_api/common_functions.h>
#include <cstdint>
#include "score_functor.h"
#include "types.h"
#include "utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_1024) inline void assign_scores_kernel_with_filter_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity, uint32_t bucket_max_size,
    uint32_t dim, __gm__ K* keys, __gm__ S* scores, uint64_t n,
    uint64_t global_epoch, const uint64_t total_thread_num,
    uint64_t system_cycle, uint32_t block_id, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;
  uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;

  if (kv_idx >= n) {
    return;
  }
  if (!buckets) {
    return;
  }
  if (!keys) {
    return;
  }

  K key{static_cast<K>(EMPTY_KEY)};

  __gm__ K* bucket_keys_ptr = buckets->keys_;

  VecD_Comp target_digests{0};
  uint32_t key_pos = {0};
  const VecD_Comp empty_digests_val = empty_digests<K>();
  for (; kv_idx < n; kv_idx += total_thread_num) {
    key = keys[kv_idx];
    bool done = false;

    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      key_pos = global_idx & (bucket_max_size - 1);
      uint64_t bkt_idx = global_idx >> max_bucket_shift;
      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys_ptr = bucket->keys_;
    } else {
      continue;
    }

    // One more loop to handle empty keys.
    constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);
    for (uint32_t offset = 0; offset < bucket_max_size + STRIDE && !done;
         offset += STRIDE) {
      uint32_t pos_cur = align_to<STRIDE>(key_pos);
      pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

      __gm__ D* digests_ptr =
          BUCKET::digests(bucket_keys_ptr, bucket_max_size, pos_cur);

      VecD_Comp probe_digests =
          *(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));
      uint32_t possible_pos = 0;
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;
      while (cmp_result != 0 && !done) {
        // NPU uses little endian,
        // and the lowest byte in register stores in the lowest address.
        uint32_t index =
            (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + index;

        __gm__ K* current_key_ptr = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K try_key = asc_atomic_cas(current_key_ptr, key,
                                             static_cast<K>(LOCKED_KEY));
        if (try_key == key) {
          key_pos = possible_pos;
          ScoreFunctor::update_without_missed(bucket_keys_ptr, bucket_max_size,
                                              key_pos, scores, kv_idx,
                                              global_epoch, system_cycle);
          done = true;
          (void)asc_atomic_exch(current_key_ptr, key);
        }
      }
      if (!done) {
        cmp_result = vcmpeq4(probe_digests, empty_digests_val);
        cmp_result &= 0x01010101;
        while (cmp_result != 0 && !done) {
          uint32_t index =
              (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;
          // 如果offset为0，并且possible_pos小于key_pos，则跳过
          // 因为key_pos是已经找到的key的位置，如果possible_pos小于key_pos，则说明possible_pos已经被处理过了
          if (offset == 0 && possible_pos < key_pos) {
            continue;
          }
          K current_key = bucket_keys_ptr[possible_pos];
          if (current_key == static_cast<K>(EMPTY_KEY)) {
            done = true;
          }
        }
      }
    }
  }
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int Strategy = -1, int32_t TILE_SIZE = 32>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_1024) inline void assign_scores_kernel_with_io_vf(
  __gm__ Bucket<K, V, S>* buckets, uint64_t capacity, uint32_t bucket_max_size,
    uint32_t dim, __gm__ K* keys, __gm__ S* scores, uint64_t n,
    uint64_t global_epoch, const uint64_t total_thread_num,
    uint64_t system_cycle, uint32_t block_id, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  if (!buckets) {
    return;
  }
  if (!keys) {
    return;
  }
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;
  auto lane_id = threadIdx.x % TILE_SIZE;
  const uint64_t N = n * TILE_SIZE;

  for (uint64_t t = block_id * blockDim.x + threadIdx.x; t < N;
       t += total_thread_num) {
    uint64_t key_idx = t / TILE_SIZE;
    K key = keys[key_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      continue;
    }
    const K hashed_key = Murmur3HashDevice(key);
    uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                         capacity_divisor_shift, capacity);
    uint32_t key_pos = global_idx & (bucket_max_size - 1);
    uint64_t bkt_idx = global_idx >> max_bucket_shift;

    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __gm__ K* bucket_keys = bucket->keys_;

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_and_lock_for_update<K, S, TILE_SIZE>(
        bucket_keys, bucket_max_size, key, key_pos, lane_id);

    if (occupy_result == OccupyResult::REFUSED) continue;
    if (occupy_result == OccupyResult::DUPLICATE) {
      if (lane_id == 0) {
        ScoreFunctor::update_without_missed(bucket_keys, bucket_max_size,
                                            key_pos, scores, key_idx,
                                            global_epoch, system_cycle);
      }
      asc_threadfence();
    }
    if (lane_id == 0) {
      (void)asc_atomic_exch(bucket_keys + key_pos, key);
    }
  }
}

template <class K, class V, class S, int Strategy = -1>
__global__ __vector__ void assign_scores_kernel_with_filter(
  __gm__ Bucket<K, V, S>* buckets, uint64_t capacity, uint32_t bucket_max_size,
    uint32_t dim, __gm__ K* keys, __gm__ S* scores, uint64_t n,
    uint64_t global_epoch, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  uint64_t system_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
  const uint64_t total_thread_num = THREAD_NUM_1024 * GetBlockNum();

  asc_vf_call<assign_scores_kernel_with_filter_vf<K, V, S, Strategy>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_1024)}, buckets, capacity,
      bucket_max_size, dim, keys, scores, n, global_epoch, total_thread_num,
      system_cycle, GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
      capacity_divisor_shift);
}

template <class K, class V, class S, int Strategy = -1>
__global__ __vector__ void assign_scores_kernel_with_io(
  __gm__ Bucket<K, V, S>* buckets, uint64_t capacity, uint32_t bucket_max_size,
    uint32_t dim, __gm__ K* keys, __gm__ S* scores, uint64_t n,
    uint64_t global_epoch, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  uint64_t system_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
  const uint64_t total_thread_num = THREAD_NUM_1024 * GetBlockNum();

  asc_vf_call<assign_scores_kernel_with_io_vf<K, V, S, Strategy>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_1024)}, buckets, capacity,
      bucket_max_size, dim, keys, scores, n, global_epoch, total_thread_num,
      system_cycle, GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
      capacity_divisor_shift);
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_ASSIGN_SCORES_KERNEL_WITH_FILTER_H_
