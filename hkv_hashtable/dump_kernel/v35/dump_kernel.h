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

#ifndef ASCENDC_DUMP_KERNEL_H_
#define ASCENDC_DUMP_KERNEL_H_

#include <kernel_operator.h>
#include <cstdint>

#include "../../../include/types.h"
#include "../../../include/utils.h"
#include "simt_api/asc_simt.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 2048;

template <class K, class V, class S, int32_t GROUP_SIZE = 16>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void dump_kernel_vf(
    GM_ADDR table_gm, GM_ADDR buckets_gm, GM_ADDR key_gm, GM_ADDR val_gm,
    GM_ADDR score_gm, const size_t offset, const size_t search_length,
    const uint64_t total_thread_num, GM_ADDR dump_counter_gm,
    uint64_t ub_shared_kvs_mem, __ubuf__ uint32_t* block_acc,
    __ubuf__ uint32_t* global_acc, uint32_t dim_in, uint32_t blockIdx) {
  if ((!table_gm) || (!buckets_gm) || (!key_gm) || (!val_gm) ||
      (!dump_counter_gm)) {
    return;
  }

  auto block_tuples =
      reinterpret_cast<__ubuf__ KVM<K, V, S>*>(ub_shared_kvs_mem);

  __gm__ Table<K, V, S>* __restrict__ table =
      reinterpret_cast<__gm__ Table<K, V, S>*>(table_gm);
  __gm__ Bucket<K, V, S>* __restrict__ buckets =
      reinterpret_cast<__gm__ Bucket<K, V, S>*>(buckets_gm);
  __gm__ K* __restrict__ d_key = reinterpret_cast<__gm__ K*>(key_gm);
  __gm__ V* __restrict__ d_val = reinterpret_cast<__gm__ V*>(val_gm);
  __gm__ S* __restrict__ d_score = reinterpret_cast<__gm__ S*>(score_gm);
  __gm__ size_t* __restrict__ d_dump_counter =
      reinterpret_cast<__gm__ size_t*>(dump_counter_gm);

  const size_t bucket_max_size{table->bucket_max_size};

  size_t tid{blockIdx * blockDim.x + threadIdx.x};
  for (; tid < search_length; tid += total_thread_num) {
    if (threadIdx.x == 0) {
      block_acc[0] = 0;
    }
    AscendC::Simt::ThreadBarrier();

    __gm__ Bucket<K, V, S>* bucket = buckets + (tid + offset) / bucket_max_size;
    __gm__ K* bucket_keys_ptr = bucket->keys_;
    __gm__ V* bucket_values_ptr = bucket->vectors;
    __gm__ S* bucket_scores_ptr = bucket->scores_;

    const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
    const K key{bucket_keys_ptr[key_idx]};

    if (!IS_RESERVED_KEY<K>(key)) {
      size_t local_index{atomicAdd(block_acc, 1u)};
      block_tuples[local_index].key = key;
      block_tuples[local_index].value = reinterpret_cast<uint64_t>(&bucket_values_ptr[key_idx * dim_in]);
      block_tuples[local_index].score = bucket_scores_ptr[key_idx];
    }
    AscendC::Simt::ThreadBarrier();

    if (threadIdx.x == 0) {
      global_acc[0] =
          atomicAdd(d_dump_counter, static_cast<size_t>(block_acc[0]));
    }
    AscendC::Simt::ThreadBarrier();

    if (block_acc[0] == 0) {
      continue;
    }

    auto cg_rank_id = threadIdx.x % GROUP_SIZE;
    auto cg_rank_id_start = threadIdx.x - cg_rank_id;
    uint64_t tuple_value_ptr = 0;
    __ubuf__ const KVM<K, V, S>& tuple{block_tuples[threadIdx.x]};
    const size_t j{global_acc[0] + threadIdx.x};
    if (threadIdx.x < block_acc[0]) {
      d_key[j] = tuple.key;

      if (d_score != nullptr) {
        d_score[j] = tuple.score;
      }
    }

    for (int32_t i = 0; i < GROUP_SIZE; i++) {
      if ((cg_rank_id_start + i) < block_acc[0]) {
        // 协程组并行写入向量值
        auto val_start = asc_shfl(j, i, GROUP_SIZE);
        uint64_t value_sync_ptr = asc_shfl(tuple.value, i, GROUP_SIZE);
        auto value_sync = reinterpret_cast<__gm__ V*>(value_sync_ptr);
        for (uint32_t k = cg_rank_id; k < dim_in; k += GROUP_SIZE) {
          __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
              L1CacheType::NON_CACHEABLE>(
            d_val + val_start * dim_in + k, value_sync[k]);
        }
      }
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_DUMP_KERNEL_H_