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

#ifndef ACCUM_OR_ASSIGN_KERNEL_HYBRID_H_
#define ACCUM_OR_ASSIGN_KERNEL_HYBRID_H_

#include <cstdint>
#include "../../include/score_functor.h"
#include "../../include/types.h"
#include "../../include/utils.h"
#include "../../include/find_utils.h"
#include "kernel_operator.h"

namespace npu {
namespace hkv {
using namespace AscendC;

/*
 * Phase-1: follows accum_or_assign_kernel_vf exactly, but replaces
 * the in-kernel value write (accum_or_assign_vector) with recording
 * the destination pointer and accum flag for phase-2.
 */
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t Strategy = -1, int32_t TILE_SIZE = 32>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void accum_or_assign_lock_key_hybrid_vf(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int32_t* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const K* keys,
    __gm__ const bool* accum_or_assigns, __gm__ const S* scores,
    S cur_score, uint64_t n, uint32_t thread_all,
    uint64_t global_epoch, uint32_t block_index,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift,
    __gm__ V* __gm__* d_dst_values, __gm__ bool* d_accum_or_assigns) {
  if (buckets == nullptr || buckets_size == nullptr ||
      keys == nullptr || accum_or_assigns == nullptr) {
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
      if (lane_id == 0) {
        d_dst_values[key_idx] = nullptr;
        d_accum_or_assigns[key_idx] = false;
      }
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
      if (lane_id == 0) {
        d_dst_values[key_idx] = nullptr;
        d_accum_or_assigns[key_idx] = false;
      }
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
        d_dst_values[key_idx] = nullptr;
        d_accum_or_assigns[key_idx] = false;
      }
      continue;
    }

    // 4. 新插入位置需要更新 bucket_size
    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        lane_id == 0) {
      atomicAdd(bucket_size, 1);
    }

    // 5. 记录目标 value 地址和 accum 标志（替代原来的 accum_or_assign_vector）
    if (lane_id == 0) {
      d_dst_values[key_idx] = bucket_vectors + key_pos * dim;
      d_accum_or_assigns[key_idx] = is_accum;
    }

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
__global__ __vector__ void accum_or_assign_lock_key_hybrid_kernel(
    __gm__ Bucket<K, V, S>* buckets, __gm__ int* buckets_size,
    uint64_t capacity, uint32_t bucket_max_size, uint32_t dim,
    __gm__ const K* keys,
    __gm__ const bool* accum_or_assigns, __gm__ const S* scores,
    uint64_t n, uint64_t global_epoch,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift,
    __gm__ V* __gm__* d_dst_values, __gm__ bool* d_accum_or_assigns) {

  const uint32_t thread_all = THREAD_NUM_512 * GetBlockNum();
  uint64_t cur_score =
      (Strategy == npu::hkv::EvictStrategyInternal::kLru ||
       Strategy == npu::hkv::EvictStrategyInternal::kEpochLru)
          ? static_cast<uint64_t>(GetSystemCycle())
          : 0;

  asc_vf_call<accum_or_assign_lock_key_hybrid_vf<K, V, S, Strategy>>(
      dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets,
      buckets_size, capacity, bucket_max_size, dim, keys,
      accum_or_assigns, scores, cur_score, n, thread_all,
      global_epoch, GetBlockIdx(), max_bucket_shift,
      capacity_divisor_magic, capacity_divisor_shift,
      d_dst_values, d_accum_or_assigns);
}

/*
 * Phase-2: write_with_accum_kernel
 * Single-pass SIMD design using 3 TQue buffers (2 VECIN + 1 VECOUT):
 * ASSIGN entries: DataCopyPad src→UB, Adds(out,src,0)→VECOUT, DataCopyPad→dst
 * ACCUM  entries: DataCopyPad src→UB + dst→UB, Add(out,src,dst), DataCopyPad→dst
*/
template <class V>
__global__ __vector__ void write_with_accum_kernel(
    uint32_t former_num, uint64_t former_core_move_num,
    uint64_t tail_core_move_num, uint32_t tile_size, uint32_t num_tiles,
    uint32_t dim, __gm__ V* values, uint64_t n,
    __gm__ V* __gm__* d_dst_values, __gm__ bool* d_accum_or_assigns) {
  uint64_t cur_block_idx = GetBlockIdx();
  uint64_t core_start_idx = 0;
  uint64_t core_move_count = 0;
  if (cur_block_idx < former_num) {
    core_start_idx = cur_block_idx * former_core_move_num;
    core_move_count = former_core_move_num;
  } else {
    core_start_idx = former_num * former_core_move_num +
                     (cur_block_idx - former_num) * tail_core_move_num;
    core_move_count = tail_core_move_num;
  }

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueueX, inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueZ;
  pipe.InitBuffer(inQueueX, DOUBLE_BUFFER, tile_size * sizeof(V));
  pipe.InitBuffer(inQueueY, DOUBLE_BUFFER, tile_size * sizeof(V));
  pipe.InitBuffer(outQueueZ, DOUBLE_BUFFER, tile_size * sizeof(V));

  AscendC::GlobalTensor<V> src_values_gm;
  AscendC::GlobalTensor<V> dst_values_gm;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};

  AscendC::LocalTensor<V> src_local, dst_local, out_local;

  src_values_gm.SetGlobalBuffer(values);

  for (uint64_t i = core_start_idx; i < core_start_idx + core_move_count; i++) {
    __gm__ V* dst_value = d_dst_values[i];
    if (dst_value == nullptr) {
      continue;
    }

    bool is_accum = d_accum_or_assigns[i];
    uint64_t src_offset = i * dim;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      uint32_t current_tile_size = (tile_idx == num_tiles - 1)
                                       ? (dim - tile_idx * tile_size)
                                       : tile_size;
      DataCopyExtParams copy_params{
          1, static_cast<uint32_t>(current_tile_size * sizeof(V)), 0, 0, 0};
      uint32_t tile_offset = tile_idx * tile_size;

      dst_values_gm.SetGlobalBuffer(dst_value + tile_offset);

      inQueueX.AllocTensor(src_local);
      AscendC::DataCopyPad(src_local,
                           src_values_gm[src_offset + tile_offset],
                           copy_params, pad_params);
      inQueueX.EnQue(src_local);
      inQueueX.DeQue(src_local);

      outQueueZ.AllocTensor(out_local);

      if (is_accum) {
        inQueueY.AllocTensor(dst_local);
        AscendC::DataCopyPad(dst_local, dst_values_gm, copy_params, pad_params);
        inQueueY.EnQue(dst_local);
        inQueueY.DeQue(dst_local);

        AscendC::Add(out_local, src_local, dst_local, current_tile_size);

        inQueueY.FreeTensor(dst_local);
      } else {
        AscendC::Adds(out_local, src_local, static_cast<V>(0), current_tile_size);
      }

      outQueueZ.EnQue<V>(out_local);
      outQueueZ.DeQue(out_local);

      AscendC::DataCopyPad(dst_values_gm, out_local, copy_params);

      inQueueX.FreeTensor(src_local);
      outQueueZ.FreeTensor(out_local);
    }
  }
}

}  // namespace hkv
}  // namespace npu

#endif  // ACCUM_OR_ASSIGN_KERNEL_HYBRID_H_
