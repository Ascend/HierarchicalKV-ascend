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
 * \file find_or_insert_ptr_kernel.h
 * \brief find_or_insert_ptr_kernel
 */

#ifndef ASCENDC_FIND_OR_INSERT_PTR_KERNEL_H_
#define ASCENDC_FIND_OR_INSERT_PTR_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "../../../include/score_functor.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_or_insert_ptr_kernel_vf(
    GM_ADDR buckets_gm, GM_ADDR buckets_size_gm, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys_gm,
    GM_ADDR value_ptrs_gm, GM_ADDR scores, GM_ADDR key_ptrs_gm, uint64_t n,
    GM_ADDR founds_gm, uint64_t global_epoch, uint64_t cur_score,
    uint32_t blockIdx, uint64_t thread_all) {
  if (!buckets_gm) {
    return;
  }
  if (!buckets_size_gm) {
    return;
  }
  if (!keys_gm) {
    return;
  }
  if (!value_ptrs_gm) {
    return;
  }
  if (!founds_gm) {
    return;
  }
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_gm);
  __gm__ V* __gm__* __restrict__ value_ptrs =
      reinterpret_cast<__gm__ V * __gm__*>(value_ptrs_gm);
  __gm__ K* __gm__* __restrict__ key_ptrs =
      reinterpret_cast<__gm__ K * __gm__*>(key_ptrs_gm);
  __gm__ bool* __restrict__ founds = reinterpret_cast<__gm__ bool*>(founds_gm);
  __gm__ S* __restrict__ scores_ptr = reinterpret_cast<__gm__ S*>(scores);
  S score{static_cast<S>(EMPTY_SCORE)};

  for (uint64_t kv_idx = blockIdx * blockDim.x + threadIdx.x; kv_idx < n;
       kv_idx += thread_all) {
    // 1. 每个线程处理一个key
    K key = keys[kv_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      founds[kv_idx] = false;
      value_ptrs[kv_idx] = nullptr;
      continue;
    }
    score = ScoreFunctor::desired_when_missed(scores_ptr, kv_idx, global_epoch, cur_score);

    // 2. 计算key的hash值 && 定位key
    const K hashed_key = Murmur3HashDevice(key);
    uint64_t global_idx =
        static_cast<uint64_t>(hashed_key % (buckets_num * bucket_capacity));
    uint32_t key_pos = global_idx % bucket_capacity;
    uint64_t bkt_idx = global_idx / bucket_capacity;

    __gm__ int32_t* bucket_size_ptr = buckets_size + bkt_idx;
    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __gm__ K* bucket_keys_ptr = bucket->keys_;
    __gm__ V* bucket_values_ptr = bucket->vectors;
    __gm__ S* bucket_scores_ptr = bucket->scores_;

    // 3. 遍历桶，找key
    bool found = false;
    uint32_t target_pos = INVALID_KEY_POS;
    for (int offset = 0; offset < bucket_capacity; offset++) {
      uint32_t current_pos = (key_pos + offset) % bucket_capacity;
      auto current_key_ptr = bucket_keys_ptr + current_pos;
      K key_val = *current_key_ptr;
      if (key_val == key) {
        // 3.1 找到现有键
        found = true;
        target_pos = current_pos;
        ScoreFunctor::update_with_digest(bucket_keys_ptr, target_pos, scores_ptr,
          kv_idx, score, bucket_capacity, get_digest<K>(key), false);
        break;
      } else if (key_val == EMPTY_KEY) {
        // 3.2 找到空位，尝试插入
        K expected = EMPTY_KEY;
        auto try_key = Simt::AtomicCas(reinterpret_cast<__gm__ K*>(current_key_ptr),
                                       expected, LOCKED_KEY);
        if (try_key == expected) {
          target_pos = current_pos;
          *current_key_ptr = key;
          ScoreFunctor::update_with_digest(bucket_keys_ptr, target_pos, scores_ptr,
            kv_idx, score, bucket_capacity, get_digest<K>(key), true);
          atomicAdd(bucket_size_ptr, 1);
          found = true;
          break;
        }
      }
    }

    // 4. 开始准入淘汰
    if (target_pos == INVALID_KEY_POS) {
      S min_score = MAX_SCORE;
      uint32_t min_pos = target_pos;
      // 4.1 遍历桶，找最小值
      for (uint32_t current_pos = 0; current_pos < bucket_capacity;
           current_pos++) {
        auto current_score = bucket_scores_ptr[current_pos];
        if (current_score < min_score && bucket_keys_ptr[current_pos] != LOCKED_KEY) {
          min_score = current_score;
          min_pos = current_pos;
        }
      }
      // 4.2 分数不足，无法准入
      if (score > min_score) {
        auto current_key_ptr = bucket_keys_ptr + min_pos;
        auto current_key = *current_key_ptr;
        if (current_key != LOCKED_KEY) {
          auto try_key = Simt::AtomicCas(current_key_ptr, current_key, LOCKED_KEY);
          // 抢占成功
          if (try_key == current_key) {
            if (min_score >= bucket_scores_ptr[min_pos]) {
              target_pos = min_pos;
              ScoreFunctor::update_with_digest(bucket_keys_ptr, target_pos, scores_ptr,
                kv_idx, score, bucket_capacity, get_digest<K>(key), true);
              *current_key_ptr = key;
            } else {
              (void)Simt::AtomicExch(current_key_ptr, current_key);
            }
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

    founds[kv_idx] = found;
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_OR_INSERT_PTR_KERNEL_H_
