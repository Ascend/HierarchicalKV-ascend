/* *
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

/* !
 * \file utils_kernel.h
 * \brief utils_kernel
 */

#ifndef ASCENDC_UITLS_KERNEL_H_
#define ASCENDC_UITLS_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 1024;

template <class V>
__simt_vf__ __aicore__
LAUNCH_BOUND(BLOCK_SIZE) inline void read_from_ptr_kernel_vf(
    GM_ADDR src_addr, GM_ADDR dst_addr,
    const size_t dim, size_t N, uint32_t blockIdx, uint32_t blockNums) {
  auto src = reinterpret_cast<const __gm__ V* const __gm__* __restrict>(src_addr);
  auto dst = reinterpret_cast<__gm__ V* __restrict >(dst_addr);

  size_t tid = (blockIdx * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * blockNums) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    if (src[vec_index]) {
      dst[vec_index * dim + dim_index] = src[vec_index][dim_index];
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_UITLS_KERNEL_H_
