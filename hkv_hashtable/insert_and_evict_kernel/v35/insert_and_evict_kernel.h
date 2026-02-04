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

#ifndef ASCENDC_INSERT_AND_EVICT_KERNEL_H_
#define ASCENDC_INSERT_AND_EVICT_KERNEL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "../../../include/score_functor.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

template <typename K = uint64_t, typename V = float, typename S = uint64_t, int Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void insert_and_evict_kernel_vf(
    GM_ADDR buckets_addr_gm, GM_ADDR buckets_size_addr_gm, uint64_t capacity,
    uint32_t bucket_capacity, uint32_t dim, GM_ADDR keys_addr_gm, GM_ADDR values_addr_gm,
    GM_ADDR scores_gm, S cur_score, GM_ADDR evicted_keys_addr_gm, GM_ADDR evicted_values_addr_gm,
    GM_ADDR evicted_scores_addr_gm, GM_ADDR d_evicted_counter_addr_gm, uint64_t n, uint64_t thread_all,
    uint64_t global_epoch, uint32_t block_index, uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  if (buckets_addr_gm == nullptr || buckets_size_addr_gm == nullptr) {
    return;
  }
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_addr_gm);
  __gm__ V* __restrict__ values = reinterpret_cast<__gm__ V*>(values_addr_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  __gm__ K* __restrict__ evicted_keys = reinterpret_cast<__gm__ K*>(evicted_keys_addr_gm);
  __gm__ V* __restrict__ evicted_values = reinterpret_cast<__gm__ V*>(evicted_values_addr_gm);
  __gm__ S* __restrict__ evicted_scores = reinterpret_cast<__gm__ S*>(evicted_scores_addr_gm);
  __gm__ uint64_t* __restrict__ d_evicted_counter = reinterpret_cast<__gm__ uint64_t*>(d_evicted_counter_addr_gm);
  S score{static_cast<S>(EMPTY_SCORE)};

  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x; kv_idx < n; kv_idx += thread_all) {
    // 1. 每个线程处理一个key
    K key = keys[kv_idx];
    if (IS_RESERVED_KEY<K>(key)) {
      continue;
    }
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                              cur_score);

    // 2. 计算key的hash值 && 定位key
    const K hashed_key = Murmur3HashDevice(key);
    uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                         capacity_divisor_shift, capacity);
    uint32_t key_pos = global_idx & (bucket_capacity - 1);
    uint64_t bkt_idx = global_idx >> max_bucket_shift;

    __gm__ int32_t* bucket_size = buckets_size + bkt_idx;
    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __gm__ K* bucket_keys = bucket->keys_;
    __gm__ V* bucket_values = bucket->vectors;
    __gm__ S* bucket_scores = bucket->scores_;

    // 3. 遍历桶，找key
    uint32_t target_pos = INVALID_KEY_POS;
    for (uint32_t offset = 0; offset < bucket_capacity; offset++) {
      uint32_t current_pos = (key_pos + offset) % bucket_capacity;
      auto current_key_ptr = bucket_keys + current_pos;
      auto current_key = *current_key_ptr;
      // 3.1 找到key
      if (current_key == key) {
        auto try_key = Simt::AtomicCas(current_key_ptr, current_key, LOCKED_KEY);
        // 抢占成功
        if (try_key == current_key) {
          target_pos = current_pos;
          ScoreFunctor::update_score_only(bucket_keys, target_pos, scores,
                                          kv_idx, score, bucket_capacity,
                                          false);
          break;
        }
        // 3.2 找到空位
      } else if (current_key == EMPTY_KEY) {
        auto try_key = Simt::AtomicCas(current_key_ptr, EMPTY_KEY, LOCKED_KEY);
        // 抢占成功
        if (try_key == EMPTY_KEY) {
          target_pos = current_pos;
          ScoreFunctor::update_with_digest(bucket_keys, target_pos, scores,
                                           kv_idx, score, bucket_capacity,
                                           get_digest<K>(key), true);
          atomicAdd(bucket_size, 1);
          break;
        }
      }
    }

    // 4. 开始准入淘汰
    bool is_evicted = false;
    uint64_t evicted_idx = 0;
    while (target_pos == INVALID_KEY_POS) {
      S min_score = MAX_SCORE;
      uint32_t min_pos = target_pos;
      // 4.1 遍历桶，找最小值
      for (uint32_t current_pos = 0; current_pos < bucket_capacity;
           current_pos++) {
        auto current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(bucket_scores + current_pos);
        auto current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(bucket_keys + current_pos);
        if (current_score < min_score && current_key != LOCKED_KEY && current_key != EMPTY_KEY) {
          min_score = current_score;
          min_pos = current_pos;
        }
      }
      // 4.2 分数不足，无法准入，淘汰设置为待插入的key
      if (score < min_score) {
        evicted_idx = atomicAdd(d_evicted_counter, 1);
        evicted_keys[evicted_idx] = key;
        if (evicted_scores != nullptr) {
          evicted_scores[evicted_idx] = score;
        }
        break;
      }
      auto current_key_ptr = bucket_keys + min_pos;
      auto current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(current_key_ptr);
      if (current_key != LOCKED_KEY && current_key != EMPTY_KEY) {
        auto try_key = Simt::AtomicCas(current_key_ptr, current_key, LOCKED_KEY);
        // 抢占成功
        if (try_key == current_key) {
          if (min_score >= __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(bucket_scores + min_pos)) {
            target_pos = min_pos;
            ScoreFunctor::update_with_digest(bucket_keys, target_pos, scores,
                                             kv_idx, score, bucket_capacity,
                                             get_digest<K>(key), true);
            if (current_key == RECLAIM_KEY) {
              atomicAdd(bucket_size, 1);
            } else {
              evicted_idx = atomicAdd(d_evicted_counter, 1);
              evicted_keys[evicted_idx] = current_key;
              if (evicted_scores != nullptr) {
                evicted_scores[evicted_idx] = min_score;
              }
              is_evicted = true;
            }
          } else {
            (void)Simt::AtomicExch(current_key_ptr, current_key);
          }
        }
      }
    }

    // 5. 抢占成功，写入value，写入evicted_values
    if (target_pos != INVALID_KEY_POS) {
      if (is_evicted) {
        for (uint32_t i = 0; i < dim; i++) {
          __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(
            evicted_values + evicted_idx * dim + i, bucket_values[target_pos * dim + i]);
        }
      }
      size_t bucket_value_start = target_pos * dim;
      size_t value_start = kv_idx * dim;
      for (uint32_t i = 0; i < dim; i++) {
        __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(
            bucket_values + bucket_value_start + i, values[value_start + i]);
      }
      __threadfence();

      // key也是原子标记位，所有key的操作必须原子化
      (void)Simt::AtomicExch(bucket_keys + target_pos, key);
    } else {
      for (uint32_t i = 0; i < dim; i++) {
        __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(
            evicted_values + evicted_idx * dim + i, values[kv_idx * dim + i]);
      }
    }
  }
}


template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t Strategy = -1, int32_t GROUP_SIZE = 16,
          int32_t EVICT_GROUP_SIZE = 16, int32_t COUNT_GROUP_SIZE = 32>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void insert_and_evict_kernel_with_digest_vf(
    GM_ADDR buckets_addr_gm, GM_ADDR buckets_size_addr_gm, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, GM_ADDR keys_addr_gm,
    GM_ADDR values_addr_gm, GM_ADDR scores_gm, S cur_score,
    GM_ADDR evicted_keys_addr_gm, GM_ADDR evicted_values_addr_gm,
    GM_ADDR evicted_scores_addr_gm, GM_ADDR d_evicted_counter_addr_gm,
    uint64_t n, uint32_t thread_all, uint64_t global_epoch,
    uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, __ubuf__ uint32_t* block_acc,
    __ubuf__ uint64_t* global_acc) {
  if (buckets_addr_gm == nullptr || buckets_size_addr_gm == nullptr) {
    return;
  }

  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_addr_gm);
  __gm__ int32_t* __restrict__ buckets_size =
      reinterpret_cast<__gm__ int32_t*>(buckets_size_addr_gm);
  __gm__ const K* __restrict__ keys =
      reinterpret_cast<__gm__ const K*>(keys_addr_gm);
  __gm__ V* __restrict__ values = reinterpret_cast<__gm__ V*>(values_addr_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  __gm__ K* __restrict__ evicted_keys =
      reinterpret_cast<__gm__ K*>(evicted_keys_addr_gm);
  __gm__ V* __restrict__ evicted_values =
      reinterpret_cast<__gm__ V*>(evicted_values_addr_gm);
  __gm__ S* __restrict__ evicted_scores =
      reinterpret_cast<__gm__ S*>(evicted_scores_addr_gm);
  __gm__ uint64_t* __restrict__ d_evicted_counter =
      reinterpret_cast<__gm__ uint64_t*>(d_evicted_counter_addr_gm);

  S score = static_cast<S>(EMPTY_SCORE);
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = 0;
  K evict_key = 0;
  S evict_score = 0;
  __gm__ K* bucket_keys = nullptr;
  uint64_t bucket_values_uintptr = 0;
  __gm__ S* bucket_scores = nullptr;
  __gm__ int32_t* bucket_size = nullptr;

  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    VecD_Comp target_digests{0};
    OccupyResult occupy_result{OccupyResult::INITIAL};

    // 1. 每个线程处理一个 key
    if (kv_idx < n) {
      key = keys[kv_idx];
      if (IS_RESERVED_KEY<K>(key)) {
        occupy_result = OccupyResult::ILLEGAL;
      }
      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                cur_score);

      // 2. 计算 key 的 hash 值 && 定位 key
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                           capacity_divisor_shift, capacity);
      key_pos = global_idx & (bucket_max_size - 1);
      uint64_t bkt_idx = global_idx >> max_bucket_shift;

      bucket_size = buckets_size + bkt_idx;
      int32_t cur_bucket_size = *bucket_size;
      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys = bucket->keys_;
      bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);
      bucket_scores = bucket->scores_;

      // 3. 使用 digest 遍历桶，找 key/空位
      for (uint32_t offset = 0; offset < bucket_max_size + STRIDE;
           offset += STRIDE) {
        if (occupy_result != OccupyResult::INITIAL) {
          break;
        }
        uint32_t pos_cur = align_to<STRIDE>(key_pos);
        pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

        __gm__ D* digests_ptr =
            BUCKET::digests(bucket_keys, bucket_max_size, pos_cur);
        VecD_Comp probe_digests =
            *reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr);

        // 3.1 遍历 digest，4 个比较
        uint32_t possible_pos = 0;
        uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index =
              (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              Simt::AtomicCas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
          // 3.2 找到 key，尝试抢占
          if (try_key == key) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            ScoreFunctor::update_score_only(bucket_keys, key_pos, scores,
                                            kv_idx, score, bucket_max_size,
                                            false);
            break;
          }
        } while (true);
        // 3.3 找到了，跳出循环
        if (occupy_result == OccupyResult::DUPLICATE) {
          break;
        } else if (cur_bucket_size == bucket_max_size) {
          // 3.4 未找到，且桶已满，进行下一波对比
          continue;
        }
        // 3.5 未找到，桶未满，找空桶
        VecD_Comp empty_digests_ = empty_digests<K>();
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        do {
          if (cmp_result == 0) {
            break;
          }
          uint32_t index =
              (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;
          if (offset == 0 && possible_pos < key_pos) {
            continue;
          }

          auto current_key_ptr = BUCKET::keys(bucket_keys, possible_pos);
          auto try_key =
              Simt::AtomicCas(current_key_ptr, static_cast<K>(EMPTY_KEY),
                              static_cast<K>(LOCKED_KEY));
          // 3.6 找到空位，尝试抢占
          if (try_key == static_cast<K>(EMPTY_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            key_pos = possible_pos;
            ScoreFunctor::update_with_digest(bucket_keys, key_pos, scores,
                                             kv_idx, score, bucket_max_size,
                                             get_digest<K>(key), true);
            atomicAdd(bucket_size, 1);
            break;
          }
        } while (true);
        // 3.7 抢占到空位，跳出循环，否则进行下一波对比
        if (occupy_result == OccupyResult::OCCUPIED_EMPTY) {
          break;
        }
      }
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    // 4. 协程组优化：准入淘汰
    // 前面查找会有 3 种结果:
    // * OccupyResult::DUPLICATE 抢占 key
    // * OccupyResult::OCCUPIED_EMPTY 抢占空位
    // * OccupyResult::INITIAL 均抢占失败，需要尝试淘汰
    auto cg_rank_id = threadIdx.x % EVICT_GROUP_SIZE;
    // 遍历组内线程，每个线程都要有可能淘汰
    for (int32_t i = 0; i < EVICT_GROUP_SIZE; i++) {
      auto res_sync = __shfl(occupy_result, i, EVICT_GROUP_SIZE);
      while (res_sync == OccupyResult::INITIAL) {
        S min_score = MAX_SCORE;
        uint32_t min_pos = key_pos;
        // 4.1 协程组并行遍历桶，找最小值
        uint64_t bucket_scores_sync = __shfl(
            reinterpret_cast<uint64_t>(bucket_scores), i, EVICT_GROUP_SIZE);
        uint64_t bucket_keys_sync = __shfl(
            reinterpret_cast<uint64_t>(bucket_keys), i, EVICT_GROUP_SIZE);
        for (uint32_t current_pos = cg_rank_id; current_pos < bucket_max_size;
             current_pos += EVICT_GROUP_SIZE) {
          S current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                  L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ S*>(bucket_scores_sync) + current_pos);
          K current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                                L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ K*>(bucket_keys_sync) + current_pos);
          if (current_score < min_score &&
              current_key != static_cast<K>(LOCKED_KEY) &&
              current_key != static_cast<K>(EMPTY_KEY)) {
            min_score = current_score;
            min_pos = current_pos;
          }
        }
        // 分治法求最小值，最终所有线程获得相同的 min_score 和 min_pos
        for (int32_t offset = EVICT_GROUP_SIZE / 2; offset > 0; offset /= 2) {
          S other_score = __shfl_xor(min_score, offset, EVICT_GROUP_SIZE);
          uint32_t other_pos = __shfl_xor(min_pos, offset, EVICT_GROUP_SIZE);
          if (other_score < min_score) {
            min_score = other_score;
            min_pos = other_pos;
          }
        }
        // 拿到了最小值和位置，后续要进行 value 搬运
        if (cg_rank_id == i) {
          // 4.2 分数不足，无法准入，淘汰设置为待插入的 key
          if (score < min_score) {
            occupy_result = OccupyResult::REFUSED;
            evict_key = key;
            evict_score = score;
          } else {
            // 4.3 分数满足，尝试准入
            auto current_key_ptr = bucket_keys + min_pos;
            auto current_key =
                __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                      L1CacheType::NON_CACHEABLE>(current_key_ptr);
            if (current_key != static_cast<K>(LOCKED_KEY) &&
                current_key != static_cast<K>(EMPTY_KEY)) {
              auto try_key = Simt::AtomicCas(current_key_ptr, current_key,
                                             static_cast<K>(LOCKED_KEY));
              // 4.4 抢占成功
              if (try_key == current_key) {
                // 4.4.1 确认分数是不是变更小
                if (__ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                          L1CacheType::NON_CACHEABLE>(bucket_scores +
                                                      min_pos) <= min_score) {
                  key_pos = min_pos;
                  ScoreFunctor::update_with_digest(
                      bucket_keys, key_pos, scores, kv_idx, score,
                      bucket_max_size, get_digest<K>(key), true);
                  if (try_key == static_cast<K>(RECLAIM_KEY)) {
                    occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
                    atomicAdd(bucket_size, 1);
                  } else {
                    occupy_result = OccupyResult::EVICT;
                    evict_key = current_key;
                    evict_score = min_score;
                  }
                } else {
                  // 4.4.2 分数变大，淘汰失败，把 key 还原回去，重新遍历
                  (void)Simt::AtomicExch(current_key_ptr, current_key);
                }
              }
              // 4.5 抢占失败，重新遍历
            }
          }
        }
        res_sync = __shfl(occupy_result, i, EVICT_GROUP_SIZE);
      }
    }
    if (threadIdx.x == 0) {
      *block_acc = 0;
    }
    AscendC::Simt::ThreadBarrier();
    // 5. 前缀和计算偏移：每个 GROUP 只做一次 atomicAdd
    auto rank = threadIdx.x % COUNT_GROUP_SIZE;

    // 5.1 判断是否需要 evict
    bool need_evict = (occupy_result == OccupyResult::REFUSED ||
                       occupy_result == OccupyResult::EVICT);
    uint32_t my_count = need_evict ? 1 : 0;

    // 5.2 计算 exclusive 前缀和
    uint32_t prefix_sum = my_count;
    for (int32_t offset = 1; offset < COUNT_GROUP_SIZE; offset *= 2) {
      uint32_t other = __shfl_up(prefix_sum, offset, COUNT_GROUP_SIZE);
      if (rank >= offset) {
        prefix_sum += other;
      }
    }
    uint32_t local_offset = prefix_sum - my_count;

    // 5.3 获取组内总数（最后一个线程的 prefix_sum）
    uint32_t group_evict_count = __shfl(prefix_sum, COUNT_GROUP_SIZE - 1, COUNT_GROUP_SIZE);

    // 5.4 组内第一个线程做 atomicAdd 获取基地址
    uint32_t group_base = 0;
    if (rank == 0 && group_evict_count > 0) {
      group_base = atomicAdd(block_acc, group_evict_count);
    }
    group_base = __shfl(group_base, 0, COUNT_GROUP_SIZE);

    AscendC::Simt::ThreadBarrier();
    if (threadIdx.x == 0) {
      *global_acc = atomicAdd(d_evicted_counter, *block_acc);
    }
    AscendC::Simt::ThreadBarrier();

    // 5.5 计算最终的 evicted_idx
    uint64_t evicted_idx = *global_acc + group_base + local_offset;

    // 5.6 写入 evicted_keys 和 evicted_scores
    if (need_evict) {
      evicted_keys[evicted_idx] = evict_key;
      if (evicted_scores != nullptr) {
        evicted_scores[evicted_idx] = evict_score;
      }
    }
    rank = threadIdx.x % GROUP_SIZE;
    // 5.7 协程组搬运 value
    for (int32_t i = 0; i < GROUP_SIZE; i++) {
      auto occupy_result_cur = __shfl(occupy_result, i, GROUP_SIZE);
      if (occupy_result_cur == OccupyResult::ILLEGAL) {
        continue;
      }

      auto kv_idx_cur = __shfl(kv_idx, i, GROUP_SIZE);
      auto evicted_idx_cur = __shfl(evicted_idx, i, GROUP_SIZE);
      auto key_pos_cur = __shfl(key_pos, i, GROUP_SIZE);
      uint64_t bucket_values_cur = __shfl(bucket_values_uintptr, i, GROUP_SIZE);

      __gm__ V* bucket_values_ptr =
          reinterpret_cast<__gm__ V*>(bucket_values_cur);

      if (occupy_result_cur != OccupyResult::REFUSED) {
        // EVICT: 先搬运旧值到 evicted_values
        if (occupy_result_cur == OccupyResult::EVICT) {
          for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
            __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(
                evicted_values + evicted_idx_cur * dim + j,
                bucket_values_ptr[key_pos_cur * dim + j]);
          }
        }
        // 搬运新值到 bucket_values
        for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
          __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(
              bucket_values_ptr + key_pos_cur * dim + j,
              values[kv_idx_cur * dim + j]);
        }
      } else {
        // REFUSED: 搬运新值到 evicted_values
        for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
          __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(
              evicted_values + evicted_idx_cur * dim + j,
              values[kv_idx_cur * dim + j]);
        }
      }
    }
    __threadfence();

    // 解锁
    if (occupy_result != OccupyResult::REFUSED &&
        occupy_result != OccupyResult::ILLEGAL) {
      // key也是原子标记位，所有key的操作必须原子化
      (void)Simt::AtomicExch(bucket_keys + key_pos, key);
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_INSERT_AND_EVICT_KERNEL_H_
