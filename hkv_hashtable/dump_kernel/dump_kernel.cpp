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
 * \file dump_kernel.cpp
 * \brief dump_kernel
 */
#include "./v35/dump_kernel.h"

#include <cstdint>

#include "kernel_operator.h"
#include "../../include/simt_vf_dispatcher.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void dump_kernel(
    GM_ADDR table, GM_ADDR buckets, GM_ADDR keys, GM_ADDR vals, GM_ADDR scores,
    const size_t offset, const size_t search_length, GM_ADDR dump_counter,
    uint32_t value_size, int32_t group_size, uint32_t dim) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  using K = uint64_t;
  using V = float;
  using S = uint64_t;
  const uint64_t total_thread_num = THREAD_NUM * GetBlockNum();

  AscendC::TPipe pipe;

  AscendC::TBuf<AscendC::TPosition::VECCALC> shared_kvs_mem;
  pipe.InitBuffer(shared_kvs_mem, THREAD_NUM * sizeof(KVM<K, V, S>));
  AscendC::LocalTensor<KVM<K, V, S>> shared_kvs_tensor =
      shared_kvs_mem.Get<KVM<K, V, S>>();
  AscendC::TBuf<AscendC::TPosition::VECCALC> block_acc;
  pipe.InitBuffer(block_acc, sizeof(uint32_t));
  AscendC::LocalTensor<uint32_t> shared_block_acc_tensor =
      block_acc.Get<uint32_t>();
  __ubuf__ uint32_t* ub_shared_block_acc_mem =
      reinterpret_cast<__ubuf__ uint32_t*>(
          shared_block_acc_tensor.GetPhyAddr());

  AscendC::TBuf<AscendC::TPosition::VECCALC> global_acc;
  pipe.InitBuffer(global_acc, sizeof(uint32_t));
  AscendC::LocalTensor<uint32_t> shared_global_acc_tensor =
      global_acc.Get<uint32_t>();
  __ubuf__ uint32_t* ub_shared_global_acc_mem =
      reinterpret_cast<__ubuf__ uint32_t*>(
          shared_global_acc_tensor.GetPhyAddr());

  DISPATCH_GROUP_SIZE(
      group_size,
      DISPATCH_VALUE_SIZE(
          value_size,
          (Simt::VF_CALL<dump_kernel_vf<K, DTYPE, S, GROUP_SIZE>>(
              Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)}, table, buckets,
              keys, vals, scores, offset, search_length, total_thread_num,
              dump_counter, shared_kvs_tensor.GetPhyAddr(),
              ub_shared_block_acc_mem, ub_shared_global_acc_mem, dim,
              GetBlockIdx()))));
}

#ifdef USE_DUMP_KERNEL_ASC
namespace npu {
namespace hkv {
constexpr uint32_t UB_SIZE = 64 * 1024;
void dump_kernel_do(uint32_t blockDim, void* stream, void* table, void* buckets,
                    void* keys, void* vals, void* scores, const size_t offset,
                    const size_t search_length, void* dump_counter,
                    uint32_t value_size, int32_t group_size, uint32_t dim) {
  dump_kernel<<<blockDim, UB_SIZE, stream>>>(
      (GM_ADDR)table, (GM_ADDR)buckets, (GM_ADDR)keys, (GM_ADDR)vals,
      (GM_ADDR)scores, offset, search_length, (GM_ADDR)dump_counter, value_size,
      group_size, dim); 
}
}  // namespace hkv
}  // namespace npu
#endif