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

#ifndef ASCENDC_REMOVE_KERNEL_H_
#define ASCENDC_REMOVE_KERNEL_H_

#include <cstdint>
#include "../../include/score_functor.h"
#include "../../include/types.h"
#include "../../include/utils.h"
#include "kernel_operator.h"
#include "simt_api/device_warp_functions.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void remove_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int32_t* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim, __gm__ K* keys,
    uint64_t n, uint32_t thread_all, uint32_t block_index,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  if (buckets == nullptr) {
    return;
  }
  if (buckets_size == nullptr) {
    return;
  }
  if (keys == nullptr) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;

  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = 0;
  __gm__ K* bucket_keys = nullptr;
  __gm__ D* bucket_digests = nullptr;
  __gm__ S* bucket_scores = nullptr;
  VecD_Comp target_digests{0};
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x; kv_idx < n;
       kv_idx += thread_all) {
    // 1. 每个线程处理一个key
    key = keys[kv_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      continue;
    }

    // 2. 计算key的hash值 && 定位key
    const K hashed_key = Murmur3HashDevice(key);
    target_digests = digests_from_hashed<K>(hashed_key);
    uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                         capacity_divisor_shift, capacity);
    key_pos = global_idx & (bucket_max_size - 1);
    uint64_t bkt_idx = global_idx >> (max_bucket_shift);

    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    bucket_keys = bucket->keys_;
    bucket_digests = bucket->digests_;
    bucket_scores = bucket->scores_;

    // 3. 遍历桶，找key/空位
    for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
         offset += STRIDE) {
      uint32_t pos_cur = align_to<STRIDE>(key_pos);
      pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

      __gm__ D* digests_ptr =
          BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
      VecD_Comp probe_digests =
          *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);
      // 3.1 遍历digest，4个比较
      uint32_t possible_pos = 0;
      uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;
      do {
        if (cmp_result == 0) {
          break;
        }
        uint32_t index = (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + index;

        // 3.2
        // 找到key，使用RECLAIM_KEY来替换它；更新key、score、digest、bucket_size
        if (bucket_keys[possible_pos] == key) {
          bucket_keys[possible_pos] = static_cast<K>(RECLAIM_KEY);
          bucket_scores[possible_pos] = EMPTY_SCORE;
          bucket_digests[possible_pos] = reclaim_digest<K>();
          atomicSub(buckets_size + bkt_idx, 1);
          break;
        }
      } while (true);
    }
  }
}

template <typename K, typename V, typename S,
          template <typename, typename> class PredFunctor>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void remove_if_kernel_vf(
    __gm__ Bucket<K, V, S>* __restrict__ buckets, __gm__ int32_t* buckets_size,
    uint64_t buckets_num, uint32_t bucket_max_size, const K pattern,
    const S threshold, __gm__ size_t* count, uint32_t thread_all,
    uint32_t block_index) {
  PredFunctor<K, S> pred;

  K key = 0;
  S score = 0;
  __gm__ K* bucket_keys = nullptr;
  __gm__ D* bucket_digests = nullptr;
  __gm__ S* bucket_scores = nullptr;
  __gm__ Bucket<K, V, S>* bucket = nullptr;
  uint32_t bucket_erase_count = 0;
  uint32_t thread_erase_count = 0;
  for (uint64_t bucket_idx = block_index * blockDim.x + threadIdx.x;
       bucket_idx < buckets_num; bucket_idx += thread_all) {
    // 每个线程处理一个桶
    bucket = buckets + bucket_idx;
    bucket_keys = bucket->keys_;
    bucket_digests = bucket->digests_;
    bucket_scores = bucket->scores_;
    bucket_erase_count = 0;
    for (uint32_t pos_cur = 0; pos_cur < bucket_max_size; pos_cur++) {
      key = bucket_keys[pos_cur];
      if (IS_RESERVED_KEY<K>(key)) {
        continue;
      }
      score = bucket_scores[pos_cur];
      if (pred(key, score, pattern, threshold)) {
        bucket_erase_count++;
        bucket_keys[pos_cur] = static_cast<K>(RECLAIM_KEY);
        bucket_scores[pos_cur] = EMPTY_SCORE;
        bucket_digests[pos_cur] = reclaim_digest<K>();
      }
    }
    thread_erase_count = thread_erase_count + bucket_erase_count;
    buckets_size[bucket_idx] = buckets_size[bucket_idx] - bucket_erase_count;
  }
  // asc_reduce_add不支持uint64_t的累加，只能使用uint32_t，但考虑到32个线程累加桶内key数超过uint32_max也不可能
  uint32_t warp_erase_count = asc_reduce_add(thread_erase_count);
  if (threadIdx.x % warpSize == 0) {
    atomicAdd(count, warp_erase_count);
  }
}

template <typename K, typename V, typename S, typename PredFunctor,
          int32_t GROUP_SIZE>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void remove_if_v2_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int32_t* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    uint32_t thread_all, uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    PredFunctor pred, __gm__ size_t* count) {
  for (uint64_t tid = block_index * blockDim.x + threadIdx.x; tid < capacity;
       tid += thread_all) {
    uint64_t bkt_idx = tid >> max_bucket_shift;
    uint64_t key_idx = tid - (bkt_idx << max_bucket_shift);

    __gm__ Bucket<K, V, S>* __restrict__ bucket = buckets + bkt_idx;

    const K key = bucket->keys_[key_idx];
    __gm__ V* value = bucket->vectors + key_idx * dim;
    const S score = bucket->scores_[key_idx];

    bool match = false;
    if (!IS_RESERVED_KEY<K>(key) && pred(key, value, score, GROUP_SIZE)) {
      match = true;
      bucket->keys_[key_idx] = static_cast<K>(RECLAIM_KEY);
      bucket->scores_[key_idx] = EMPTY_SCORE;
      bucket->digests_[key_idx] = reclaim_digest<K>();
      if (bucket_max_size < warpSize) {
        atomicSub(buckets_size + bkt_idx, 1);
      }
    }
    uint32_t vote = asc_ballot(match);
    int32_t warp_count = AscendC::Simt::Popc(vote);
    if (threadIdx.x % warpSize == 0) {
      atomicAdd(count, warp_count);
      if (bucket_max_size >= warpSize) {
        atomicSub(buckets_size + bkt_idx, warp_count);
      }
    }
  }
}

template <class K, class V, class S>
__global__ __vector__ void remove_kernel(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int32_t* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim, __gm__ K* keys,
    uint64_t n, uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();

  Simt::VF_CALL<remove_kernel_vf<K, V, S>>(
      Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, buckets_size,
      capacity, bucket_max_size, dim, keys, n, thread_all, GetBlockIdx(),
      max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift);
}

template <class K, class V, class S,
          template <typename, typename> class PredFunctor>
__global__ __vector__ void remove_if_kernel(__gm__ Bucket<K, V, S>* buckets,
                                            __gm__ int32_t* buckets_size,
                                            uint64_t buckets_num,
                                            uint32_t bucket_max_size,
                                            const K pattern, const S threshold,
                                            __gm__ size_t* count) {
  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();

  asc_vf_call<remove_if_kernel_vf<K, V, S, PredFunctor>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, buckets_size,
      buckets_num, bucket_max_size, pattern, threshold, count, thread_all,
      GetBlockIdx());
}

template <class K, class V, class S, class PredFunctor, int32_t GROUP_SIZE>
__global__ __vector__ void remove_if_v2_kernel(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int32_t* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift, PredFunctor pred, __gm__ size_t* count) {
  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();

  asc_vf_call<remove_if_v2_kernel_vf<K, V, S, PredFunctor, GROUP_SIZE>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, buckets_size,
      capacity, bucket_max_size, dim, thread_all, GetBlockIdx(),
      max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift, pred,
      count);
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_REMOVE_KERNEL_H_
