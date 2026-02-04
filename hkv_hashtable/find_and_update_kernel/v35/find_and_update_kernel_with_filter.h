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
 * \file find_and_update_kernel_with_filter.h
 * \brief find_and_update_kernel_with_filter
 */

#ifndef ASCENDC_FIND_AND_UPDATE_KERNEL_WITH_FILTER_H_
#define ASCENDC_FIND_AND_UPDATE_KERNEL_WITH_FILTER_H_

#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/score_functor.h"
#include "../../../include/types.h"
#include "../../../include/utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int Strategy = -1>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void find_and_update_kernel_with_filter_vf(
    GM_ADDR buckets_gm, uint64_t capacity, uint32_t bucket_capacity,
    uint32_t dim, GM_ADDR keys_gm, GM_ADDR value_ptrs_gm, GM_ADDR scores_gm,
    GM_ADDR founds_gm, uint64_t n, bool update_score, uint64_t global_epoch,
    const uint64_t total_thread_num, uint64_t system_cycle, uint32_t block_id,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  using BUCKET = Bucket<K, V, S>;
  using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;
  uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;
  K key{static_cast<K>(EMPTY_KEY)};
  S score{static_cast<S>(EMPTY_SCORE)};

  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ K* __restrict__ keys = reinterpret_cast<__gm__ K*>(keys_gm);
  __gm__ V* __gm__* __restrict__ value_ptrs =
      reinterpret_cast<__gm__ V * __gm__*>(value_ptrs_gm);
  __gm__ S* __restrict__ scores = reinterpret_cast<__gm__ S*>(scores_gm);
  __gm__ bool* __restrict__ founds = reinterpret_cast<__gm__ bool*>(founds_gm);

  __gm__ K* bucket_keys_ptr = buckets->keys_;
  __gm__ V* bucket_values_ptr = buckets->vectors;
  OccupyResult occupy_result{OccupyResult::INITIAL};

  VecD_Comp target_digests{0};
  uint32_t key_pos = {0};
  const VecD_Comp empty_digests_ = empty_digests<K>();
  for (; kv_idx < n; kv_idx += total_thread_num) {
    key = keys[kv_idx];
    occupy_result = OccupyResult::INITIAL;
    score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch,
                                              system_cycle);
    bool done = false;

    if (!IS_RESERVED_KEY<K>(key)) {
      const K hashed_key = Murmur3HashDevice(key);
      target_digests = digests_from_hashed<K>(hashed_key);
      const uint64_t global_idx = get_global_idx(
          hashed_key, capacity_divisor_magic, capacity_divisor_shift, capacity);
      key_pos = global_idx & (bucket_capacity - 1);
      const uint64_t bkt_idx = global_idx >> max_bucket_shift;
      __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
      bucket_keys_ptr = bucket->keys_;
      bucket_values_ptr = bucket->vectors;
    } else {
      occupy_result = OccupyResult::ILLEGAL;
      done = true;
    }

    // One more loop to handle empty keys.
    constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);
    for (uint32_t offset = 0; offset < bucket_capacity + STRIDE && !done;
         offset += STRIDE) {
      uint32_t pos_cur = align_to<STRIDE>(key_pos);
      pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

      __gm__ D* digests_ptr =
          BUCKET::digests(bucket_keys_ptr, bucket_capacity, pos_cur);

      const VecD_Comp probe_digests =
          *(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));
      uint32_t possible_pos = 0;
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
      cmp_result &= 0x01010101;
      while (cmp_result != 0 && !done) {
        // NPU uses little endian,
        // and the lowest byte in register stores in the lowest address.
        const uint32_t index =
            (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
        cmp_result &= (cmp_result - 1);
        possible_pos = pos_cur + index;

        __gm__ K* current_key_ptr = BUCKET::keys(bucket_keys_ptr, possible_pos);
        K try_key = AscendC::Simt::AtomicCas(current_key_ptr, key, LOCKED_KEY);
        if (try_key == key) {
          occupy_result = OccupyResult::DUPLICATE;
          key_pos = possible_pos;
          ScoreFunctor::update_score_only(bucket_keys_ptr, key_pos, scores,
                                          kv_idx, score, bucket_capacity,
                                          false);
          done = true;
          (void)AscendC::Simt::AtomicExch(current_key_ptr, key);
        }
      }
      if (!done) {
        cmp_result = vcmpeq4(probe_digests, empty_digests_);
        cmp_result &= 0x01010101;
        while (cmp_result != 0 && !done) {
          const uint32_t index =
              (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;
          // 如果offset为0，并且possible_pos小于key_pos，则跳过
          // 因为key_pos是已经找到的key的位置，如果possible_pos小于key_pos，则说明possible_pos已经被处理过了
          if (offset == 0 && possible_pos < key_pos) {
            continue;
          }
          K current_key = bucket_keys_ptr[possible_pos];
          if (current_key == static_cast<K>(EMPTY_KEY)) {
            occupy_result = OccupyResult::OCCUPIED_EMPTY;
            done = true;
          }
        }
      }
    }

    // WRITE_BACK: 写回逻辑
    bool found_ = occupy_result == OccupyResult::DUPLICATE;
    if (founds) {
      founds[kv_idx] = found_;
    }
    if (found_) {
      if (scores) {
        score = *BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
        scores[kv_idx] = score;
      }
      value_ptrs[kv_idx] = bucket_values_ptr + key_pos * dim;
    } else {
      value_ptrs[kv_idx] = nullptr;
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_AND_UPDATE_KERNEL_WITH_FILTER_H_