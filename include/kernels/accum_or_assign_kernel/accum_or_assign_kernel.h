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

#ifndef ACCUM_OR_ASSIGN_KERNEL_H_
#define ACCUM_OR_ASSIGN_KERNEL_H_

#include <cstdint>
#include "score_functor.h"
#include "types.h"
#include "utils.h"
#include "find_utils.h"
#include "kernel_operator.h"

namespace npu {
namespace hkv {
using namespace AscendC;

/*
 * accum_or_assign_vector: TILE_SIZE threads cooperate on value write.
 * When is_accum=true:  dst[i] += src[i]  (accumulate delta)
 * When is_accum=false: dst[i] = src[i]   (assign value)
 * dim is the original dimension in V elements.
 */
template <typename V, int32_t TILE_SIZE>
__forceinline__ __simt_callee__ void accum_or_assign_vector(
    __gm__ const V* src, __gm__ V* dst,
    const bool is_accum, const uint32_t dim, const uint32_t lane_id) {
  for (uint32_t i = lane_id; i < dim; i += TILE_SIZE) {
    V val = src[i];
    if (is_accum) {
      val += __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                   L1CacheType::NON_CACHEABLE>(dst + i);
    }
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
          L1CacheType::NON_CACHEABLE>(dst + i, val);
  }
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t Strategy = -1, int32_t TILE_SIZE = 32>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void accum_or_assign_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int32_t* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const K* keys, __gm__ const V* values,
    __gm__ const bool* accum_or_assigns, __gm__ const S* scores,
    S cur_score, uint64_t n, uint32_t thread_all,
    uint64_t global_epoch, uint32_t block_index,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  if (buckets == nullptr || buckets_size == nullptr ||
      keys == nullptr || values == nullptr || accum_or_assigns == nullptr) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  auto lane_id = threadIdx.x % TILE_SIZE;
  const uint64_t N = n * TILE_SIZE;

  for (uint64_t t = block_index * blockDim.x + threadIdx.x;
       t < N; t += thread_all) {
    uint64_t key_idx = t / TILE_SIZE;

    K key = keys[key_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      continue;
    }
    S score = ScoreFunctor::desired_when_missed(scores, key_idx, global_epoch,
                                                cur_score);
    const bool is_accum = accum_or_assigns[key_idx];

    // 1. 计算 hash 值，定位 bucket
    const K hashed_key = Murmur3HashDevice(key);
    uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                         capacity_divisor_shift, capacity);
    uint32_t key_pos = global_idx & (bucket_max_size - 1);
    uint64_t bkt_idx = global_idx >> max_bucket_shift;

    __gm__ int32_t* bucket_size = buckets_size + bkt_idx;
    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __gm__ K* bucket_keys = bucket->keys_;
    __gm__ V* bucket_vectors = bucket->vectors;
    __gm__ S* bucket_scores = bucket->scores_;

    // 2. 32 线程协作查找 key / 空位 / 淘汰
    K evicted_key = static_cast<K>(EMPTY_KEY);
    OccupyResult occupy_result;
    do {
      occupy_result = find_and_lock<K, S, TILE_SIZE>(
          bucket_keys, bucket_scores, bucket_max_size,
          key, score, key_pos, evicted_key, lane_id);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) {
      continue;
    }
    // 3. 条件拒绝：
    //    - 想累加但 key 不存在(新位置) → 撤销
    //    - 想赋值但 key 已存在(DUPLICATE) → 撤销
    if ((is_accum && occupy_result != OccupyResult::DUPLICATE) ||
        (!is_accum && occupy_result == OccupyResult::DUPLICATE)) {
      if (lane_id == 0) {
        K restore_key = evicted_key;
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          restore_key = static_cast<K>(EMPTY_KEY);
        } else if (occupy_result == OccupyResult::OCCUPIED_RECLAIMED) {
          restore_key = static_cast<K>(RECLAIM_KEY);
        } else if (occupy_result == OccupyResult::DUPLICATE) {
          restore_key = key;
        }
        (void)asc_atomic_exch(bucket_keys + key_pos, restore_key);
      }
      continue;
    }

    // 4. 新插入位置需要更新 bucket_size
    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        lane_id == 0) {
      atomicAdd(bucket_size, 1);
    }

    // 5. 协作写入 value（累加或赋值）
    accum_or_assign_vector<V, TILE_SIZE>(
        values + key_idx * dim,
        bucket_vectors + key_pos * dim,
        is_accum, dim, lane_id);

    asc_threadfence();

    // 6. 更新 score、digest，解锁 key
    if (lane_id == 0) {
      ScoreFunctor::update_with_digest(
          bucket_keys, key_pos, scores, key_idx, score,
          bucket_max_size, get_digest<K>(key),
          (occupy_result != OccupyResult::DUPLICATE));
      asc_threadfence();
      (void)asc_atomic_exch(bucket_keys + key_pos, key);
    }
  }
}

template <class K, class V, class S, int Strategy = -1>
__global__ __vector__ void accum_or_assign_kernel(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const K* keys, __gm__ const V* values,
    __gm__ const bool* accum_or_assigns, __gm__ const S* scores,
    uint64_t n, uint64_t global_epoch,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {

  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();
  uint64_t cur_score =
      (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
       Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
          ? static_cast<uint64_t>(GetSystemCycle())
          : 0;

  asc_vf_call<accum_or_assign_kernel_vf<K, V, S, Strategy>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets,
      buckets_size, capacity, bucket_max_size, dim, keys, values,
      accum_or_assigns, scores, cur_score, n, thread_all,
      global_epoch, GetBlockIdx(), max_bucket_shift,
      capacity_divisor_magic, capacity_divisor_shift);
}

}  // namespace hkv
}  // namespace npu

#endif  // ACCUM_OR_ASSIGN_KERNEL_H_
