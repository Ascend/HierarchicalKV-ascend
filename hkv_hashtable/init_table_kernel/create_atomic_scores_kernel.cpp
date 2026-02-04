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
 * \file create_atomic_scores_kernel.cpp
 * \brief create_atomic_scores_kernel
 */

#include <kernel_operator.h>
#include <cstddef>
#include <cstdint>
#include "./v35/init_table_kernel.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

using K = uint64_t;
using S = uint64_t;
using V = float;

extern "C" __global__ __aicore__ void create_atomic_scores_kernel(
    GM_ADDR buckets_gm, const size_t start, const size_t end,
    const size_t bucket_max_size, uint32_t value_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const uint64_t thread_all = SCORES_THREAD_NUM * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      (Simt::VF_CALL<create_atomic_scores_vf<K, DTYPE, S>>(
          Simt::Dim3{static_cast<uint32_t>(SCORES_THREAD_NUM)}, buckets_gm, start,
          end, bucket_max_size, GetBlockIdx(), thread_all)));
}
