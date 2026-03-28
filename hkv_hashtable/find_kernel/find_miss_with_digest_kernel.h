/* *
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#ifndef ASCENDC_FIND_MISS_WITH_DIGEST_KERNEL_H_
#define ASCENDC_FIND_MISS_WITH_DIGEST_KERNEL_H_

#include <cstdint>
#include <kernel_operator.h>
#include "../../include/find_utils.h"
#include "../../include/simt_vf_dispatcher.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename VecV = int32_t, int32_t COUNT_GROUP_SIZE = 32>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void find_miss_with_digest_kernel_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_capacity, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    __gm__ S* scores, __gm__ K* missed_keys, __gm__ int* missed_indices,
    __gm__ int* missed_size, uint64_t n, const uint64_t total_thread_num,
    uint32_t block_id, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, __ubuf__ uint32_t* block_acc,
    __ubuf__ uint64_t* global_acc, int32_t group_size) {
  if (buckets == nullptr || keys == nullptr || values == nullptr ||
      missed_keys == nullptr || missed_indices == nullptr ||
      missed_size == nullptr) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  __gm__ VecV* values_vec = reinterpret_cast<__gm__ VecV*>(values);

  for (uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += total_thread_num) {
    VecD_Comp target_digests{0};
    bool found = false;
    bool is_miss = false;
    K miss_key = 0;
    uint32_t target_pos = INVALID_KEY_POS;
    uint64_t bucket_values_uintptr = 0;

    // 1、每个线程处理一个key
    if (kv_idx < n) {
      const K key = ldg_l2nc_l1c(keys + kv_idx);
      if (!IS_RESERVED_KEY<K>(key)) {
        // 2、计算key哈希 && 定位桶位置
        const K hashed_key = Murmur3HashDevice(key);
        target_digests = digests_from_hashed<K>(hashed_key);
        uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                             capacity_divisor_shift, capacity);
        uint32_t key_pos = global_idx & (bucket_capacity - 1);
        uint64_t bkt_idx = global_idx >> max_bucket_shift;

        __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
        bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);

        // 3、遍历桶查找key
        OccupyResult result = find_without_lock<K, V, S>(
            bucket, key, key_pos, target_digests, target_pos, bucket_capacity);
        found = (result == OccupyResult::DUPLICATE);

        // 4、设置输出信息
        if (found) {
          if (scores != nullptr) {
            scores[kv_idx] =
                *(BUCKET::scores(bucket->keys_, bucket_capacity, target_pos));
          }
        } else {
          is_miss = true;
          miss_key = key;
        }
      }
    }

    // 5、前缀和计算偏移：每个 COUNT_GROUP_SIZE 组只做一次 atomicAdd
    if (threadIdx.x == 0) {
      *block_acc = 0;
    }
    AscendC::Simt::ThreadBarrier();

    auto rank = threadIdx.x % COUNT_GROUP_SIZE;
    uint32_t my_count = is_miss ? 1U : 0U;

    // 5.1 warp 内 inclusive 前缀和
    uint32_t prefix_sum = my_count;
    for (int32_t offset = 1; offset < COUNT_GROUP_SIZE; offset *= 2) {
      uint32_t other = asc_shfl_up(prefix_sum, offset, COUNT_GROUP_SIZE);
      if (rank >= offset) {
        prefix_sum += other;
      }
    }
    uint32_t local_offset = prefix_sum - my_count;

    // 5.2 获取组内总数
    uint32_t group_miss_count =
        asc_shfl(prefix_sum, COUNT_GROUP_SIZE - 1, COUNT_GROUP_SIZE);

    // 5.3 组内第一个线程做 atomicAdd 获取 block 级基地址
    uint32_t group_base = 0;
    if (rank == 0 && group_miss_count > 0) {
      group_base = atomicAdd(block_acc, group_miss_count);
    }
    group_base = asc_shfl(group_base, 0, COUNT_GROUP_SIZE);

    // 5.4 block 内 thread 0 做一次全局 atomicAdd
    AscendC::Simt::ThreadBarrier();
    if (threadIdx.x == 0) {
      *global_acc = static_cast<uint64_t>(
          atomicAdd(missed_size, static_cast<int>(*block_acc)));
    }
    AscendC::Simt::ThreadBarrier();

    // 5.5 计算最终 missed_idx 并写入
    if (is_miss) {
      uint64_t missed_idx = *global_acc + group_base + local_offset;
      missed_keys[missed_idx] = miss_key;
      missed_indices[missed_idx] = static_cast<int>(kv_idx);
    }

    // 6、协程组协作搬运 value（命中时从 bucket 拷到输出）
    auto cg_rank_id = threadIdx.x % group_size;
    for (int32_t i = 0; i < group_size; i++) {
      bool found_sync = asc_shfl(found, i, group_size);
      if (found_sync) {
        auto kv_idx_sync = asc_shfl(kv_idx, i, group_size);
        auto target_pos_sync = asc_shfl(target_pos, i, group_size);
        uint64_t bucket_values_sync =
            asc_shfl(bucket_values_uintptr, i, group_size);
        auto value_start = kv_idx_sync * dim;
        auto bucket_value_start = target_pos_sync * dim;
        for (uint32_t j = cg_rank_id; j < dim; j += group_size) {
          values_vec[value_start + j] =
              __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                    L1CacheType::NON_CACHEABLE>(
                  reinterpret_cast<__gm__ VecV*>(bucket_values_sync) +
                  bucket_value_start + j);
        }
      }
    }
  }
}

template <typename K, typename V, typename S>
__global__ __vector__ void find_miss_with_digest_kernel(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_capacity, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    __gm__ S* scores, __gm__ K* missed_keys, __gm__ int* missed_indices,
    __gm__ int* missed_size, uint64_t n,
    uint32_t value_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> block_acc_buf;
  pipe.InitBuffer(block_acc_buf, sizeof(uint32_t));
  AscendC::LocalTensor<uint32_t> shared_block_acc_tensor =
      block_acc_buf.Get<uint32_t>();
  __ubuf__ uint32_t* ub_shared_block_acc_mem =
      reinterpret_cast<__ubuf__ uint32_t*>(
          shared_block_acc_tensor.GetPhyAddr());

  AscendC::TBuf<AscendC::TPosition::VECCALC> global_acc_buf;
  pipe.InitBuffer(global_acc_buf, sizeof(uint64_t));
  AscendC::LocalTensor<uint64_t> shared_global_acc_tensor =
      global_acc_buf.Get<uint64_t>();
  __ubuf__ uint64_t* ub_shared_global_acc_mem =
      reinterpret_cast<__ubuf__ uint64_t*>(
          shared_global_acc_tensor.GetPhyAddr());

  const uint64_t thread_all = THREAD_NUM_512 * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      (asc_vf_call<find_miss_with_digest_kernel_vf<K, V, S, DTYPE>>(
          dim3{static_cast<uint32_t>(THREAD_NUM_512), 1, 1}, buckets,
          capacity, bucket_capacity, dim, keys, values, scores,
          missed_keys, missed_indices, missed_size, n, thread_all,
          GetBlockIdx(), max_bucket_shift, capacity_divisor_magic,
          capacity_divisor_shift, n_align_warp, ub_shared_block_acc_mem,
          ub_shared_global_acc_mem, group_size)));
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_FIND_MISS_WITH_DIGEST_KERNEL_H_
