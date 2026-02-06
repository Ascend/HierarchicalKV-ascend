/* *
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * \file find_or_insert_ptr_kernel_v2.h
 * \brief find_or_insert_ptr_kernel_v2
 */

#ifndef ASCENDC_FIND_OR_INSERT_PTR_KERNEL_V2_H_
#define ASCENDC_FIND_OR_INSERT_PTR_KERNEL_V2_H_

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
          int Strategy = -1, int32_t EVICT_GROUP_SIZE = 16>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_or_insert_ptr_kernel_vf_v2(
    GM_ADDR buckets_gm, GM_ADDR buckets_size_gm, uint64_t buckets_num,
    uint32_t bucket_max_size, uint32_t dim, GM_ADDR keys_gm,
    GM_ADDR value_ptrs_gm, GM_ADDR scores_gm, GM_ADDR key_ptrs_gm, uint64_t n,
    GM_ADDR founds_gm, uint64_t global_epoch, uint64_t cur_score,
    uint32_t blockIdx, uint64_t thread_all, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, uint64_t capacity) {
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
  using BUCKET = Bucket<K, V, S>;
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
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  S score = static_cast<S>(EMPTY_SCORE);
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  uint32_t key_pos = 0;
  K key = 0;
  __gm__ K* bucket_keys = nullptr;
  __gm__ V* bucket_values = nullptr;
  __gm__ S* bucket_scores = nullptr;
  __gm__ int32_t* bucket_size = nullptr;
  for (uint64_t kv_idx = blockIdx * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    VecD_Comp target_digests{0};
    OccupyResult occupy_result{OccupyResult::INITIAL};
    // 1. 每个线程处理一个key
    if (kv_idx < n) {
      key = keys[kv_idx];
      if (IS_RESERVED_KEY<K>(key)) {
        occupy_result = OccupyResult::ILLEGAL;
        founds[kv_idx] = false;
        value_ptrs[kv_idx] = nullptr;
      } else {
        score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                                  cur_score);

        // 2. 计算key的hash值 && 定位key
        const K hashed_key = Murmur3HashDevice(key);
        target_digests = digests_from_hashed<K>(hashed_key);
        uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                            capacity_divisor_shift, capacity);
        key_pos = global_idx & (bucket_max_size - 1);
        uint64_t bkt_idx = global_idx >> (max_bucket_shift);

        bucket_size = buckets_size + bkt_idx;
        int32_t cur_bucket_size = *bucket_size;
        __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
        bucket_keys = bucket->keys_;
        bucket_values = bucket->vectors;
        bucket_scores = bucket->scores_;

        // 3. 遍历桶，找key/空位
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
          // 3.1 遍历digest，4个比较
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
            // 3.2 找到key，尝试抢占
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
            // 3.4 未找到，且桶已满，进行下一波对比
          } else if (cur_bucket_size == bucket_max_size) {
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
      }
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    // 前面查找会有3种结果
    // * OccupyResult::DUPLICATE 抢占key
    // * OccupyResult::OCCUPIED_EMPTY 抢占空位
    // * OccupyResult::INITIAL 均抢占失败
    // 4. 开始准入淘汰
    auto cg_rank_id = threadIdx.x % EVICT_GROUP_SIZE;
    // 遍历组内线程，每个线程都要有可能淘汰
    for (int32_t i = 0; i < EVICT_GROUP_SIZE; i++) {
      auto res_sync = __shfl(occupy_result, i, EVICT_GROUP_SIZE);
      while (res_sync == OccupyResult::INITIAL) {
        S min_score = MAX_SCORE;
        uint32_t min_pos = key_pos;
        // 4.1 遍历桶，找最小值
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
        // 分治法求最小值，最终所有线程获得相同的min_score和min_pos
        for (int32_t offset = EVICT_GROUP_SIZE / 2; offset > 0; offset /= 2) {
          S other_score = __shfl_xor(min_score, offset, EVICT_GROUP_SIZE);
          uint32_t other_pos = __shfl_xor(min_pos, offset, EVICT_GROUP_SIZE);
          if (other_score < min_score) {
            min_score = other_score;
            min_pos = other_pos;
          }
        }
        // 拿到了最小值和位置，后续要进行value搬运，每个线程要维护自己的occupy_result，key_pos
        if (cg_rank_id == i) {
          // 4.2 分数不足，无法准入
          if (score < min_score) {
            occupy_result = OccupyResult::REFUSED;
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
                  }
                } else {
                  // 4.4.2 分数变大，淘汰失败，把key还原回去，重新遍历
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

    if (occupy_result == OccupyResult::ILLEGAL) {
      continue;
    }

    // 5. 抢占成功，写入value
    if (occupy_result == OccupyResult::REFUSED) {
      value_ptrs[kv_idx] = nullptr;
    } else {
      value_ptrs[kv_idx] = bucket_values + key_pos * dim;
      auto key_address = BUCKET::keys(bucket_keys, key_pos);
      *key_address = key;
    }
    founds[kv_idx] = occupy_result == OccupyResult::DUPLICATE;
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_OR_INSERT_PTR_KERNEL_V2_H_
