/* *
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#ifndef ASCENDC_CONTAINS_KERNEL_H_
#define ASCENDC_CONTAINS_KERNEL_H_

#include <cstdint>
#include <kernel_operator.h>
#include "find_utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_1024) inline void contains_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_capacity, __gm__ K* keys, __gm__ bool* founds, uint64_t n,
    const uint64_t total_thread_num, uint32_t block_id,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  if (buckets == nullptr || keys == nullptr || founds == nullptr) {
    return;
  }

  for (uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;
       kv_idx < n; kv_idx += total_thread_num) {
    bool found = false;

    const K key = ldg_l2nc_l1c(keys + kv_idx);
    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      VecD_Comp target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      uint32_t key_pos = global_idx & (bucket_capacity - 1);
      uint64_t bkt_idx = global_idx >> max_bucket_shift;

      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;

      uint32_t target_pos = INVALID_KEY_POS;
      OccupyResult result = find_without_lock<K, V, S>(
          bucket, key, key_pos, target_digests, target_pos, bucket_capacity);
      found = (result == OccupyResult::DUPLICATE);
    }

    founds[kv_idx] = found;
  }
}

template <typename K, typename V, typename S>
__global__ __vector__ void contains_kernel(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_capacity, __gm__ K* keys, __gm__ bool* founds, uint64_t n,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {

  const uint64_t thread_all = THREAD_NUM_1024 * GetBlockNum();

  asc_vf_call<contains_kernel_vf<K, V, S>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_1024), 1, 1}, buckets, capacity,
      bucket_capacity, keys, founds, n, thread_all, GetBlockIdx(),
      max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift);
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_CONTAINS_KERNEL_H_
