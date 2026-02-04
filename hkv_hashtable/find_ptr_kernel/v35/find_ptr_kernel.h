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

#ifndef ASCENDC_FIND_PTR_KERNEL_H_
#define ASCENDC_FIND_PTR_KERNEL_H_

#include <cstdint>
#include <kernel_operator.h>
#include "../../../include/types.h"
#include "../../../include/utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_ptr_kernel_vf(
    GM_ADDR buckets_gm, uint64_t buckets_num, uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys_gm,
    GM_ADDR value_ptrs_gm, GM_ADDR scores_gm, GM_ADDR founds_gm, uint64_t n, const uint64_t total_thread_num,
    uint64_t global_epoch, uint32_t blockIdx) {
  if (buckets_gm == nullptr || keys_gm == nullptr || value_ptrs_gm == nullptr) {
    return;
  }

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_gm);
  __gm__ V* __gm__* __restrict__ value_ptrs = 
      reinterpret_cast<__gm__ V * __gm__*>(value_ptrs_gm);
  __gm__ S* __restrict__ scores =
      reinterpret_cast<__gm__ S *>(scores_gm);
  __gm__ bool* __restrict__ founds =
      reinterpret_cast<__gm__ bool*>(founds_gm);

  for (uint64_t kv_idx = blockIdx * blockDim.x + threadIdx.x; kv_idx < n; kv_idx += total_thread_num) {
    // 1、每个线程处理一个key, 读取键值
    K key = keys[kv_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      if (founds != nullptr) {
        founds[kv_idx] = false;
      }
      value_ptrs[kv_idx] = nullptr;
      continue;
    }
    // 2、计算key哈希 && 定位桶位置
    const K hashed_key = Murmur3HashDevice(key);
    uint64_t global_idx =
        static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
    uint32_t key_pos = global_idx % bucket_capacity;
    uint64_t bkt_idx = global_idx / bucket_capacity;

    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __gm__ K* bucket_keys_ptr = bucket->keys_;
    __gm__ V* bucket_values_ptr = bucket->vectors;
    __gm__ S* bucket_scores_ptr = bucket->scores_;

    // 3、遍历桶查找key
    bool found = false;
    uint32_t target_pos = INVALID_KEY_POS;
    for (uint32_t offset = 0; offset < bucket_capacity; offset++) {
      uint32_t current_pos = (key_pos + offset) % bucket_capacity;
      auto current_key_ptr = bucket_keys_ptr + current_pos;
      K current_key_val = *current_key_ptr;
      // 找到现有键
      if (current_key_val == key) {
        found = true;
        target_pos = current_pos;
        break;
      } else if (current_key_val == EMPTY_KEY) {
        // 开放寻址，如果为空直接退出
        break;
      }
    }

    // 4、设置输出
    if (found) {
      value_ptrs[kv_idx] = bucket_values_ptr + target_pos * dim;
      if (scores != nullptr) {
        scores[kv_idx] = bucket_scores_ptr[target_pos];
      }
    } else {
      value_ptrs[kv_idx] = nullptr;
    }
    if (founds != nullptr) {
      founds[kv_idx] = found;
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_PTR_KERNEL_H_
