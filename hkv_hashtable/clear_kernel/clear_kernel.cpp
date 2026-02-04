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

#include "./v35/clear_kernel.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void clear_kernel(GM_ADDR buckets,
                                                   GM_ADDR buckets_size,
                                                   size_t bucket_max_size,
                                                   size_t table_capacity,
                                                   uint32_t value_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = THREAD_NUM * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      (Simt::VF_CALL<clear_kernel_vf<uint64_t, DTYPE, uint64_t>>(
          Simt::Dim3{THREAD_NUM}, buckets, buckets_size, bucket_max_size,
          table_capacity, thread_all, GetBlockIdx())));
}
