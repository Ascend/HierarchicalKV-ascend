/* *
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

#ifndef ASCENDC_UNLOCK_KEYS_KERNEL_H_
#define ASCENDC_UNLOCK_KEYS_KERNEL_H_

#include <cstdint>
#include "types.h"
#include <simt_api/common_functions.h>
#include "kernel_operator.h"

namespace npu {
namespace hkv {
using namespace AscendC;

/**
 * @brief 解锁keys kernel
 *
 * 使用lock_keys_kernel锁定的key，通过unlock_keys_kernel解锁
 * 将LOCKED_KEY替换为实际key值
 *
 * @param n key的数量
 * @param locked_key_ptrs 锁定的key指针数组
 * @param keys 实际key值数组
 * @param succeededs 解锁是否成功的标志数组
 */
template <typename K = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void unlock_keys_kernel_vf(
    uint64_t n, __gm__ K* __gm__* locked_key_ptrs,
    __gm__ const K* keys, __gm__ bool* succeededs,
    uint64_t thread_all, uint32_t block_index) {
  if (!locked_key_ptrs || !keys) {
    return;
  }

  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n; kv_idx += thread_all) {
    __gm__ K* locked_key_ptr = locked_key_ptrs[kv_idx];
    bool flag = false;

    if (locked_key_ptr != nullptr) {
      // 读取当前锁定的key值
      __gm__ K* locked_key_addr = locked_key_ptr;
      K locked_key = *locked_key_addr;
      K expected_key = static_cast<K>(LOCKED_KEY);
      K key = keys[kv_idx];

      // 只有当当前值等于LOCKED_KEY时才进行解锁
      if (locked_key == expected_key) {
        *locked_key_addr = key;
        flag = true;
      } else {
        // 已经被其他线程修改，解锁失败
        flag = false;
      }
    } else {
      // 指针为空，解锁失败
      flag = false;
    }

    if (succeededs != nullptr) {
      succeededs[kv_idx] = flag;
    }
  }
}

template <typename K>
__global__ __vector__ void unlock_keys_kernel(
    uint64_t n, __gm__ K* __gm__* locked_key_ptrs,
    __gm__ const K* keys, __gm__ bool* succeededs) {
  const uint64_t thread_all = THREAD_NUM_512 * GetBlockNum();

  asc_vf_call<unlock_keys_kernel_vf<K>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, n,
      locked_key_ptrs, keys, succeededs, thread_all, GetBlockIdx());
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_UNLOCK_KEYS_KERNEL_H_