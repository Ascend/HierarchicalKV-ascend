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

#ifndef ASCENDC_LOCK_KEYS_KERNEL_H_
#define ASCENDC_LOCK_KEYS_KERNEL_H_

#include <cstdint>
#include <simt_api/common_functions.h>
#include "../../include/score_functor.h"
#include "../../include/types.h"
#include "../../include/utils.h"
#include "kernel_operator.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <typename K = uint64_t, typename V = float, typename S = uint64_t, int32_t Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void lock_keys_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, __gm__ const K* keys,
    __gm__ K* __gm__* locked_keys_ptrs, __gm__ bool* succeed,
    __gm__ const S* scores, S cur_score, S global_epoch, uint64_t n,
    uint64_t thread_all, uint32_t block_index,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  if (!buckets && !keys && !locked_keys_ptrs && !succeed) {
    return;
  }

  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = static_cast<K>(EMPTY_KEY);
  S score = static_cast<S>(EMPTY_SCORE);
  __gm__ K* bucket_keys_ptr = nullptr;
  uint64_t bkt_idx = 0;

  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x; kv_idx < n; kv_idx += thread_all) {
    OccupyResult occupy_result{OccupyResult::INITIAL};
    VecD_Comp target_digests{0};
    bool found_ = false;

    key = keys[kv_idx];
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch, cur_score);

    if (IS_RESERVED_KEY<K>(key)) {
      occupy_result = OccupyResult::ILLEGAL;
    } else {
      // 1. 计算key的hash值 && 定位key
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic, capacity_divisor_shift,
                                            buckets_num * bucket_capacity);
      key_pos = global_idx & (bucket_capacity - 1);
      bkt_idx = global_idx >> max_bucket_shift;

      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys_ptr = bucket->keys_;

      // 2. 遍历桶查找key
      for (uint32_t offset = 0; offset < bucket_capacity + STRIDE; offset += STRIDE) {
        if (occupy_result != OccupyResult::INITIAL) {
          break;
        }
        uint32_t pos_cur = align_to<STRIDE>(key_pos);
        pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

        __gm__ D* digests_ptr = BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);
        VecD_Comp probe_digests = *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

        // 2.1 向量化比较，检查是否与目标摘要匹配。按字节进行向量比较，如果相等，将结果中对应字节设置为0xff
        uint32_t possible_pos = 0;
        uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
        cmp_result &= 0x01010101;

        // 处理匹配的摘要
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index = (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;

          // 2.2 找到key，尝试抢占为LOCKED_KEY
          auto current_key_ptr = bucket_keys_ptr + possible_pos;
          auto current_key = *current_key_ptr;
          if (current_key == key) {
            // 使用原子交换尝试锁定
            K expected_key = key;
            K desired_key = static_cast<K>(LOCKED_KEY);
            K original_key = asc_atomic_cas(current_key_ptr, expected_key, desired_key);
            if (original_key == key) {
              // 锁定成功
              occupy_result = OccupyResult::DUPLICATE;
              key_pos = possible_pos;
              // 更新score
              ScoreFunctor::update_with_digest(bucket_keys_ptr, key_pos, scores,
                                                static_cast<int32_t>(kv_idx), score,
                                                bucket_capacity,
                                                get_digest<K>(key), false);
            }
            break;
          }
        } while (true);

        // 2.3 找到了，跳出循环
        if (occupy_result == OccupyResult::DUPLICATE) {
          break;
        }
      }
    }

    found_ = (occupy_result == OccupyResult::DUPLICATE);
    if (found_) {
      locked_keys_ptrs[kv_idx] = bucket_keys_ptr + key_pos;
    } else {
      locked_keys_ptrs[kv_idx] = nullptr;
    }
    if (succeed != nullptr) {
      succeed[kv_idx] = found_;
    }
  }
}

template <typename K, typename V, typename S, int Strategy = -1>
__global__ __vector__ void lock_keys_kernel(
    __gm__ Bucket<K, V, S>* buckets, uint64_t buckets_num,
    uint32_t bucket_capacity, uint32_t dim, __gm__ const K* keys,
    __gm__ K* __gm__* locked_keys_ptrs, __gm__ bool* succeed,
    __gm__ const S* scores, uint64_t global_epoch, uint64_t n,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  const uint64_t thread_all = THREAD_NUM_512 * GetBlockNum();

  uint64_t cur_score =
      (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
       Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
          ? static_cast<uint64_t>(GetSystemCycle())
          : 0;

  asc_vf_call<lock_keys_kernel_vf<K, V, S, Strategy>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, buckets_num,
      bucket_capacity, dim, keys, locked_keys_ptrs, succeed, scores, cur_score,
      static_cast<S>(global_epoch), n, thread_all, GetBlockIdx(),
      max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift);
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_LOCK_KEYS_KERNEL_H_