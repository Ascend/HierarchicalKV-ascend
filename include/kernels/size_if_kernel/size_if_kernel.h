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

#ifndef ASCENDC_SIZE_IF_KERNEL_H_
#define ASCENDC_SIZE_IF_KERNEL_H_

#include <cstdint>
#include "score_functor.h"
#include "types.h"
#include "utils.h"
#include "kernel_operator.h"
#include "simt_api/device_warp_functions.h"
#include <simt_api/common_functions.h>

namespace npu {
namespace hkv {
using namespace AscendC;

/**
 * @brief Size_if kernel that counts elements matching a predicate.
 *
 * This kernel traverses all elements in the hash table and counts those
 * that match the given predicate (pattern, threshold).
 *
 * @tparam K Key type
 * @tparam V Value type
 * @tparam S Score type
 * @tparam PredFunctor Predicate functor template
 */
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          template <typename, typename> class PredFunctor>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void size_if_kernel_vf(
    __gm__ Bucket<K, V, S>* __restrict__ buckets, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    const K pattern, const S threshold, __gm__ size_t* count,
    uint32_t thread_all, uint32_t block_index) {
  PredFunctor<K, S> pred;

  uint32_t local_count = 0;

  for (uint64_t tid = block_index * blockDim.x + threadIdx.x; tid < capacity;
       tid += thread_all) {
    uint64_t bkt_idx = tid >> max_bucket_shift;
    uint64_t key_idx = tid - (bkt_idx << max_bucket_shift);

    __gm__ Bucket<K, V, S>* __restrict__ bucket = buckets + bkt_idx;
    const K key = bucket->keys_[key_idx];
    const S score = bucket->scores_[key_idx];

    // Check if key is not reserved and matches predicate
    if (!IS_RESERVED_KEY<K>(key) && pred(key, score, pattern, threshold)) {
      local_count++;
    }
  }

  // Use warp reduce to count, then atomic add to global counter
  uint32_t warp_count = asc_reduce_add(local_count);
  if (threadIdx.x % warpSize == 0) {
    atomicAdd(count, static_cast<size_t>(warp_count));
  }
}

/**
 * @brief Global kernel launcher for size_if
 *
 * @tparam K Key type
 * @tparam V Value type
 * @tparam S Score type
 * @tparam PredFunctor Predicate functor template
 */
template <class K, class V, class S,
          template <typename, typename> class PredFunctor>
__global__ __vector__ void size_if_kernel(
    __gm__ Bucket<K, V, S>* __restrict__ buckets, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    const K pattern, const S threshold, __gm__ size_t* count) {
  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();

  asc_vf_call<size_if_kernel_vf<K, V, S, PredFunctor>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, capacity,
      bucket_max_size, max_bucket_shift, capacity_divisor_magic,
      capacity_divisor_shift, pattern, threshold, count, thread_all,
      GetBlockIdx());
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_SIZE_IF_KERNEL_H_