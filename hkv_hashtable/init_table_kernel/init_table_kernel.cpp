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
 * \file int_table_kernel.cpp
 * \brief int_table_kernel
 */

#include <kernel_operator.h>
#include <cstdint>
#include "./v35/init_table_kernel.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void allocate_bucket_vectors_kernel(
    GM_ADDR buckets_gm, const size_t index, GM_ADDR address_gm,
    uint32_t value_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  DISPATCH_VALUE_SIZE(
      value_size,
      (Simt::VF_CALL<allocate_bucket_vectors_vf<uint64_t, DTYPE, uint64_t>>(
          Simt::Dim3{static_cast<uint32_t>(1)}, buckets_gm, index,
          address_gm)));
}