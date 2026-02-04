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

#ifndef TEST_SCORE_FUNCTOR_KERNEL_H_
#define TEST_SCORE_FUNCTOR_KERNEL_H_

#include <cstdint>
#include "kernel_operator.h"
#include "../../../include/types.h"
#include "../../../include/score_functor.h"

#define LAUNCH_BOUND(N) __attribute__((cce_launch_bounds(N)))

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t TEST_THREAD_NUM = 128;

// 调度不同策略的 VF
#define DISPATCH_STRATEGY(vf_template, ...) \
  if (strategy == EvictStrategyInternal::kLru) { \
    Simt::VF_CALL<vf_template<uint64_t, float, uint64_t, EvictStrategyInternal::kLru>>( \
        Simt::Dim3{TEST_THREAD_NUM}, __VA_ARGS__); \
  } else if (strategy == EvictStrategyInternal::kLfu) { \
    Simt::VF_CALL<vf_template<uint64_t, float, uint64_t, EvictStrategyInternal::kLfu>>( \
        Simt::Dim3{TEST_THREAD_NUM}, __VA_ARGS__); \
  } else if (strategy == EvictStrategyInternal::kEpochLru) { \
    Simt::VF_CALL<vf_template<uint64_t, float, uint64_t, EvictStrategyInternal::kEpochLru>>( \
        Simt::Dim3{TEST_THREAD_NUM}, __VA_ARGS__); \
  } else if (strategy == EvictStrategyInternal::kEpochLfu) { \
    Simt::VF_CALL<vf_template<uint64_t, float, uint64_t, EvictStrategyInternal::kEpochLfu>>( \
        Simt::Dim3{TEST_THREAD_NUM}, __VA_ARGS__); \
  } else if (strategy == EvictStrategyInternal::kCustomized) { \
    Simt::VF_CALL<vf_template<uint64_t, float, uint64_t, EvictStrategyInternal::kCustomized>>( \
        Simt::Dim3{TEST_THREAD_NUM}, __VA_ARGS__); \
  }

// 1. 测试 desired_when_missed
template <typename K, typename V, typename S, int Strategy>
__simt_vf__ __aicore__ LAUNCH_BOUND(TEST_THREAD_NUM)
inline void test_desired_when_missed_vf(
    GM_ADDR input_scores_gm, int key_idx, S epoch, S cur_cycle,
    GM_ADDR output_gm) {

  using SF = ScoreFunctor<K, V, S, Strategy>;
  __gm__ const S* input_scores = input_scores_gm != 0 
      ? reinterpret_cast<__gm__ const S*>(input_scores_gm) : nullptr;
  __gm__ S* output = reinterpret_cast<__gm__ S*>(output_gm);

  if (Simt::GetThreadIdx() == 0) {
    S result = SF::desired_when_missed(input_scores, key_idx, epoch, cur_cycle);
    *output = result;
  }
}

// 2. 测试 update (带 BUCKET* 参数)
template <typename K, typename V, typename S, int Strategy>
__simt_vf__ __aicore__ LAUNCH_BOUND(TEST_THREAD_NUM)
inline void test_update_vf(
    GM_ADDR bucket_gm, int key_pos, GM_ADDR input_scores_gm, int key_idx,
    S desired_score, bool new_insert) {

  using SF = ScoreFunctor<K, V, S, Strategy>;
  using BUCKET = Bucket<K, V, S>;

  __gm__ BUCKET* bucket = reinterpret_cast<__gm__ BUCKET*>(bucket_gm);
  __gm__ const S* input_scores = input_scores_gm != 0 
      ? reinterpret_cast<__gm__ const S*>(input_scores_gm) : nullptr;

  if (Simt::GetThreadIdx() == 0) {
    SF::update(bucket, key_pos, input_scores, key_idx, desired_score, new_insert);
  }
}

// 3. 测试 update_with_digest
template <typename K, typename V, typename S, int Strategy>
__simt_vf__ __aicore__ LAUNCH_BOUND(TEST_THREAD_NUM)
inline void test_update_with_digest_vf(
    GM_ADDR bucket_keys_gm, uint32_t key_pos, GM_ADDR input_scores_gm,
    uint32_t key_idx, S desired_score, uint32_t bucket_capacity,
    D digest, bool new_insert) {

  using SF = ScoreFunctor<K, V, S, Strategy>;

  __gm__ K* bucket_keys = reinterpret_cast<__gm__ K*>(bucket_keys_gm);
  __gm__ const S* input_scores = input_scores_gm != 0 
      ? reinterpret_cast<__gm__ const S*>(input_scores_gm) : nullptr;

  if (Simt::GetThreadIdx() == 0) {
    SF::update_with_digest(bucket_keys, key_pos, input_scores, key_idx,
                           desired_score, bucket_capacity, digest, new_insert);
  }
}

// 4. 测试 update_without_missed (带 BUCKET* 参数)
template <typename K, typename V, typename S, int Strategy>
__simt_vf__ __aicore__ LAUNCH_BOUND(TEST_THREAD_NUM)
inline void test_update_without_missed_bucket_vf(
    GM_ADDR bucket_gm, int key_pos, GM_ADDR input_scores_gm, int key_idx,
    S epoch, S cur_cycle) {

  using SF = ScoreFunctor<K, V, S, Strategy>;
  using BUCKET = Bucket<K, V, S>;

  __gm__ BUCKET* bucket = reinterpret_cast<__gm__ BUCKET*>(bucket_gm);
  __gm__ const S* input_scores = input_scores_gm != 0 
      ? reinterpret_cast<__gm__ const S*>(input_scores_gm) : nullptr;

  if (Simt::GetThreadIdx() == 0) {
    SF::update_without_missed(bucket, key_pos, input_scores, key_idx,
                              epoch, cur_cycle);
  }
}

// 5. 测试 update_without_missed (带 K* 指针参数)
template <typename K, typename V, typename S, int Strategy>
__simt_vf__ __aicore__ LAUNCH_BOUND(TEST_THREAD_NUM)
inline void test_update_without_missed_ptr_vf(
    GM_ADDR bucket_keys_gm, uint32_t bucket_capacity, uint32_t key_pos,
    GM_ADDR input_scores_gm, int key_idx, S epoch, S cur_cycle) {

  using SF = ScoreFunctor<K, V, S, Strategy>;

  __gm__ K* bucket_keys = reinterpret_cast<__gm__ K*>(bucket_keys_gm);
  __gm__ const S* input_scores = input_scores_gm != 0 
      ? reinterpret_cast<__gm__ const S*>(input_scores_gm) : nullptr;

  if (Simt::GetThreadIdx() == 0) {
    SF::update_without_missed(bucket_keys, bucket_capacity, key_pos,
                              input_scores, key_idx, epoch, cur_cycle);
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // TEST_SCORE_FUNCTOR_KERNEL_H_