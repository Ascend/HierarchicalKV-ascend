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
 * \file find_and_update_kernel.h
 * \brief find_and_update_kernel
 */

#ifndef ASCENDC_FIND_AND_UPDATE_KERNEL_H_
#define ASCENDC_FIND_AND_UPDATE_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/score_functor.h"
#include "../../../include/types.h"
#include "../../../include/utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_and_update_kernel_vf(
    GM_ADDR buckets_gm, uint64_t capacity, uint32_t bucket_capacity,
    uint32_t dim, GM_ADDR keys_gm, GM_ADDR value_ptrs_gm, GM_ADDR scores_gm,
    GM_ADDR founds_gm, uint64_t n, bool update_score, uint64_t global_epoch,
    const uint64_t total_thread_num, uint64_t system_cycle, uint32_t block_id,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;

  if (kv_idx >= n) {
    return;
  }
  if (!buckets_gm) {
    return;
  }
  if (!keys_gm) {
    return;
  }
  if (!value_ptrs_gm) {
    return;
  }
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      (__gm__ Bucket<K, V, S>*)buckets_gm;
  __gm__ const K* __restrict__ keys = (__gm__ const K*)keys_gm;
  __gm__ V* __gm__* __restrict__ value_ptrs = (__gm__ V * __gm__*)value_ptrs_gm;
  __gm__ S* __restrict__ scores = (__gm__ S*)scores_gm;
  __gm__ bool* __restrict__ founds = (__gm__ bool*)founds_gm;
  S score{static_cast<S>(EMPTY_SCORE)};

  __gm__ K* bucket_keys_ptr = buckets->keys_;
  __gm__ V* bucket_values_ptr = buckets->vectors;
  __gm__ S* bucket_scores_ptr = buckets->scores_;

  for (; kv_idx < n; kv_idx += total_thread_num) {
    uint32_t target_pos = INVALID_KEY_POS;
    bool found = false;
    K key = keys[kv_idx];
    if (!IS_RESERVED_KEY<K>(key)) {
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                system_cycle);
      // 计算哈希和桶位置
      const K hashed_key = Murmur3HashDevice(key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      uint32_t key_pos = global_idx & (bucket_capacity - 1);
      uint64_t bkt_idx = global_idx >> max_bucket_shift;

      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys_ptr = bucket->keys_;
      bucket_values_ptr = bucket->vectors;
      bucket_scores_ptr = bucket->scores_;

      // 查找并更新逻辑（简化版）
      target_pos = key_pos;
      uint32_t offset = 0;
      for (; offset < bucket_capacity; offset++) {
        uint32_t current_pos = (key_pos + offset) & (bucket_capacity - 1);
        auto current_key_ptr = bucket_keys_ptr + current_pos;
        auto current_key = *current_key_ptr;
        if (current_key == static_cast<K>(EMPTY_KEY)) {
          break;
        }
        if (current_key == key) {
          auto try_key =
              Simt::AtomicCas(current_key_ptr, current_key, LOCKED_KEY);
          // 抢占成功
          if (try_key == current_key) {
            target_pos = current_pos;
            found = true;
            ScoreFunctor::update_score_only(bucket_keys_ptr, target_pos, scores,
                                            kv_idx, score, bucket_capacity,
                                            false);
            (void)Simt::AtomicExch(current_key_ptr, key);
            if (scores) {
              scores[kv_idx] = bucket_scores_ptr[target_pos];
            }
            break;
          }
        }
      }
    }

    // 设置输出
    if (found) {
      value_ptrs[kv_idx] = bucket_values_ptr + target_pos * dim;
    } else {
      value_ptrs[kv_idx] = nullptr;
    }
    if (founds) {
      founds[kv_idx] = found;
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_AND_UPDATE_KERNEL_H_