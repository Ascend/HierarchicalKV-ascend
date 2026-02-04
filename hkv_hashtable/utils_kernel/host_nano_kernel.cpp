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
 * \file host_nano_kernel.cpp
 * \brief host_nano_kernel
 */

#include <kernel_operator.h>
#include <cstdint>
#include "./v35/utils_kernel.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void host_nano_kernel(__gm__ uint64_t* d_clk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  *d_clk = static_cast<uint64_t>(GetSystemCycle());
}
