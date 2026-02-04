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

#include "./v35/find_ptr_with_digest_kernel.h"
#include "kernel_operator.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void find_ptr_with_digest_kernel(
    GM_ADDR buckets, uint64_t capacity, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys, GM_ADDR value_ptrs,
    GM_ADDR scores, GM_ADDR founds, uint64_t n, uint64_t global_epoch, uint32_t value_size,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = THREAD_NUM * GetBlockNum();

  DISPATCH_VALUE_SIZE(
    value_size,
    (Simt::VF_CALL<find_ptr_with_digest_kernel_vf<uint64_t, DTYPE, uint64_t>>(
        Simt::Dim3{static_cast<uint32_t>(THREAD_NUM), 1, 1}, buckets,
        capacity, buckets_num, bucket_capacity, dim, keys, value_ptrs, scores, founds,
        n, thread_all, global_epoch, GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
        capacity_divisor_shift)));
}
