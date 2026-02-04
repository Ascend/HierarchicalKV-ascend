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
 * \file init_table_kernel.h
 * \brief init_table_kernel
 */

#ifndef ASCENDC_INIT_TABLE_KERNEL_H_
#define ASCENDC_INIT_TABLE_KERNEL_H_

#include <kernel_operator.h>
#include <cstddef>
#include <cstdint>
#include "../../../include/types.h"
#include "../../../include/utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

template <class K, class V, class S>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void allocate_bucket_vectors_vf(
    GM_ADDR buckets_gm, const size_t index, GM_ADDR address_gm) {
  auto buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
  __gm__ V* address = (__gm__ V*)address_gm;
  buckets[index].vectors = address;
}

template <class K, class V, class S>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void allocate_bucket_others_vf(
    GM_ADDR buckets_gm, size_t total_size_per_bucket, size_t num_of_buckets,
    const int start_index, GM_ADDR address, const uint32_t reserve_size,
    const size_t bucket_max_size) {
  auto buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
  for (size_t step = 0; step < num_of_buckets; step++) {
    size_t index = start_index + step;
    buckets[index].digests_ = reinterpret_cast<__gm__ D*>(address);
    buckets[index].keys_ =
        reinterpret_cast<__gm__ K*>(buckets[index].digests_ + reserve_size);
    buckets[index].scores_ =
        reinterpret_cast<__gm__ S*>(buckets[index].keys_ + bucket_max_size);
    address += total_size_per_bucket;
  }
}

template <class K, class V, class S>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void get_bucket_others_address_vf(
    GM_ADDR buckets_gm, const int index, GM_ADDR address_gm) {
  auto buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
  auto address = reinterpret_cast<__gm__ uint8_t*__gm__* >(address_gm);

  *address = reinterpret_cast<__gm__ uint8_t*>(buckets[index].digests_);
}

template <class K, class V, class S>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void create_atomic_keys_vf(
    GM_ADDR buckets_gm, const size_t start,
    const size_t end, const size_t bucket_max_size, uint32_t blockIdx) {
  auto buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);

  size_t tid = (blockIdx * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    for (size_t i = 0; i < bucket_max_size; i++) {
      buckets[start + tid].digests_[i] = empty_digest<K>();
      buckets[start + tid].keys_[i] = static_cast<K>(EMPTY_KEY);
    }
  }
}

constexpr uint32_t SCORES_THREAD_NUM = 2048;

template <class K, class V, class S>
__simt_vf__ __aicore__
LAUNCH_BOUND(SCORES_THREAD_NUM) inline void create_atomic_scores_vf(
    GM_ADDR buckets_gm, const size_t start, const size_t end,
    const size_t bucket_max_size, uint32_t blockIdx, uint64_t thread_all) {
  auto buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);

  for (size_t tid = (block_idx * blockDim.x) + threadIdx.x; start + tid < end;
       tid += thread_all) {
    for (size_t i = 0; i < bucket_max_size; i++) {
      buckets[start + tid].scores_[i] = static_cast<S>(EMPTY_SCORE);
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_INIT_TABLE_KERNEL_H_
