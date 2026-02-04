/**
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

#ifndef ASCENDC_CLEAR_KERNEL_H_
#define ASCENDC_CLEAR_KERNEL_H_

#include "kernel_operator.h"
#include "../../../include/types.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void clear_kernel_vf(
    GM_ADDR buckets_addr_gm, GM_ADDR buckets_size_addr_gm,
    size_t bucket_max_size, size_t table_capacity, uint64_t thread_all,
    uint32_t block_index) {
  if (buckets_addr_gm == nullptr) {
    return;
  }

  if (buckets_size_addr_gm == nullptr) {
    return;
  }

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);

  size_t t_id = block_index * blockDim.x + threadIdx.x;
  for (size_t t = t_id; t < table_capacity; t += thread_all) {
    size_t bkt_idx = t / bucket_max_size;
    size_t key_idx = t % bucket_max_size;
    auto bucket = buckets + bkt_idx;
    // 1. 桶内key置空
    buckets[bkt_idx].digests_[key_idx] = empty_digest<K>();
    buckets[bkt_idx].keys_[key_idx] = EMPTY_KEY;

    // 2. 桶大小清0
    if (key_idx == 0) {
      buckets_size[bkt_idx] = 0;
    }
  }
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_CLEAR_KERNEL_H_

