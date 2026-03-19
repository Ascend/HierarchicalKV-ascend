/**
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

#ifndef ASCENDC_TRAVERSE_KERNEL_H_
#define ASCENDC_TRAVERSE_KERNEL_H_

#include "../../include/score_functor.h"
#include "../../include/simt_vf_dispatcher.h"
#include "../../include/types.h"
#include "../../include/utils.h"
#include "kernel_operator.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <typename K, typename V, typename S, class ExecutionFunc,
          int32_t GROUP_SIZE>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_1024) inline void traverse_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, uint32_t max_bucket_size, uint32_t dim,
    uint64_t n, uint64_t first, ExecutionFunc f, uint32_t thread_all,
    uint32_t block_index) {
  uint64_t tid = block_index * blockDim.x + threadIdx.x;
  for (uint64_t i = tid; i < n; i += thread_all) {
    uint64_t bkt_idx = (i + first) / max_bucket_size;
    uint64_t key_idx = (i + first) % max_bucket_size;

    __gm__ Bucket<K, V, S>* __restrict__ bucket = buckets + bkt_idx;

    const K key = bucket->keys_[key_idx];
    __gm__ S* score = bucket->scores_ + key_idx;
    __gm__ V* value = bucket->vectors + key_idx * dim;

    f(key, value, score, GROUP_SIZE);
  }
}

template <class K, class V, class S, class ExecutionFunc, int32_t GROUP_SIZE>
__global__ __vector__ void traverse_kernel(__gm__ Bucket<K, V, S>* buckets,
                                           uint32_t max_bucket_size,
                                           uint32_t dim, uint64_t n,
                                           uint64_t first, ExecutionFunc f) {
  const uint64_t thread_all = THREAD_NUM_1024 * GetBlockNum();

  Simt::VF_CALL<traverse_kernel_vf<K, V, S, ExecutionFunc, GROUP_SIZE>>(
      Simt::Dim3{THREAD_NUM_1024}, buckets, max_bucket_size, dim, n, first, f,
      thread_all, GetBlockIdx());
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_TRAVERSE_KERNEL_H_
