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

/*!
 * \file find_or_insert_ptr_kernel_v2.cpp
 * \brief find_or_insert_ptr_kernel_v2
 */

#include "./v35/find_or_insert_ptr_kernel_v2.h"
#include <cstdint>
#include "kernel_operator.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void find_or_insert_ptr_kernel_v2(
    GM_ADDR buckets, GM_ADDR buckets_size, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys, GM_ADDR value_ptrs,
    GM_ADDR scores, GM_ADDR key_ptrs, uint64_t n, GM_ADDR founds,
    uint64_t global_epoch, int32_t evict_strategy, uint32_t value_size,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, uint64_t n_align_warp, uint64_t capacity) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = THREAD_NUM * GetBlockNum();
  uint64_t cur_score = (evict_strategy == npu::hkv::EvictStrategyInternal::kLru ||
                        evict_strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
                           ? static_cast<uint64_t>(GetSystemCycle())
                           : 0;

  DISPATCH_VALUE_SIZE(
      value_size,
      DISPATCH_EVICT_STRATEGY(
          evict_strategy,
          (Simt::VF_CALL<find_or_insert_ptr_kernel_vf_v2<uint64_t, DTYPE, uint64_t,
                                                      STRATEGY>>(
              Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)}, buckets,
              buckets_size, buckets_num, bucket_capacity, dim, keys, value_ptrs,
              scores, key_ptrs, n, founds, global_epoch, cur_score,
              GetBlockIdx(), thread_all, max_bucket_shift, capacity_divisor_magic,
              capacity_divisor_shift, n_align_warp, capacity))));
}
