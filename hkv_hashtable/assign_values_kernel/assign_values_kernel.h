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

#ifndef ASCENDC_ASSIGN_VALUES_KERNEL_H_
#define ASCENDC_ASSIGN_VALUES_KERNEL_H_

#include <simt_api/common_functions.h>
#include <cstdint>
#include "../../include/score_functor.h"
#include "../../include/simt_vf_dispatcher.h"
#include "../../include/types.h"
#include "../../include/utils.h"
#include "kernel_operator.h"

namespace npu {
namespace hkv {
using namespace AscendC;
template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          typename VecV = int32_t>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_512) inline void assign_values_kernel_with_digest_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, __gm__ K* keys,
    __gm__ void* values_addr_gm, uint64_t n, uint32_t thread_all,
    uint32_t block_index, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {
  if (!buckets) {
    return;
  }
  if (!keys) {
    return;
  }
  if (!values_addr_gm) {
    return;
  }
  using BUCKET = Bucket<K, V, S>;
  K key{static_cast<K>(EMPTY_KEY)};
  S score{static_cast<S>(EMPTY_SCORE)};

  __gm__ VecV* __restrict__ values =
      reinterpret_cast<__gm__ VecV*>(values_addr_gm);

  __gm__ K* bucket_keys_ptr = buckets->keys_;
  OccupyResult occupy_result{OccupyResult::INITIAL};

  VecD_Comp target_digests{0};
  uint32_t key_pos = {0};
  const VecD_Comp empty_digests_ = empty_digests<K>();
  uint64_t bucket_values_uintptr = 0;
  for (uint64_t kv_idx = block_index * blockDim.x + threadIdx.x;
       kv_idx < n_align_warp; kv_idx += thread_all) {
    if (kv_idx < n) {
      key = keys[kv_idx];
      occupy_result = OccupyResult::INITIAL;
      bool done = false;

      if (!IS_RESERVED_KEY<K>(key)) {
        const K hashed_key = Murmur3HashDevice(key);
        target_digests = digests_from_hashed<K>(hashed_key);
        const uint64_t global_idx =
            get_global_idx(hashed_key, capacity_divisor_magic,
                           capacity_divisor_shift, capacity);
        key_pos = global_idx & (bucket_max_size - 1);
        const uint64_t bkt_idx = global_idx >> max_bucket_shift;
        __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
        bucket_keys_ptr = bucket->keys_;
        bucket_values_uintptr = reinterpret_cast<uint64_t>(bucket->vectors);
      } else {
        occupy_result = OccupyResult::ILLEGAL;
        done = true;
      }

      // One more loop to handle empty keys.
      constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);
      for (uint32_t offset = 0; offset < bucket_max_size + STRIDE && !done;
           offset += STRIDE) {
        uint32_t pos_cur = align_to<STRIDE>(key_pos);
        pos_cur = (pos_cur + offset) & (bucket_max_size - 1);

        __gm__ D* digests_ptr =
            BUCKET::digests(bucket_keys_ptr, bucket_max_size, pos_cur);

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
              (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
          cmp_result &= (cmp_result - 1);
          possible_pos = pos_cur + index;

          __gm__ K* current_key_ptr =
              BUCKET::keys(bucket_keys_ptr, possible_pos);
          K try_key =
              asc_atomic_cas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
          if (try_key == key) {
            occupy_result = OccupyResult::DUPLICATE;
            key_pos = possible_pos;
            done = true;
          }
        }
        if (!done) {
          cmp_result = vcmpeq4(probe_digests, empty_digests_);
          cmp_result &= 0x01010101;
          while (cmp_result != 0 && !done) {
            const uint32_t index =
                (__ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
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
    } else {
      occupy_result = OccupyResult::ILLEGAL;
    }

    auto cg_rank_id = threadIdx.x % group_size;
    for (int32_t i = 0; i < group_size; i++) {
      auto res_sync = asc_shfl(occupy_result, i, group_size);
      if (res_sync == OccupyResult::DUPLICATE) {
        auto kv_idx_sync = asc_shfl(kv_idx, i, group_size);
        auto value_start = kv_idx_sync * dim;
        auto key_pos_sync = asc_shfl(key_pos, i, group_size);
        uint64_t value_ddr_sync =
            asc_shfl(bucket_values_uintptr, i, group_size);
        auto bucket_value_start = key_pos_sync * dim;
        for (uint32_t j = cg_rank_id; j < dim; j += group_size) {
          __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(
              reinterpret_cast<__gm__ VecV*>(value_ddr_sync) +
                  bucket_value_start + j,
              values[value_start + j]);
        }
      }
    }
    asc_threadfence();
    if (occupy_result == OccupyResult::DUPLICATE) {
      (void)asc_atomic_exch(bucket_keys_ptr + key_pos, key);
    }
  }
}

template <typename V, int32_t TILE_SIZE>
__forceinline__ __simt_callee__ void copy_vector(__gm__ const V* src, __gm__ V* dst,
                                            const uint32_t dim,
                                            const uint32_t lane_id) {
  for (uint32_t i = lane_id; i < dim; i += TILE_SIZE) {
    V val = src[i];
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst + i, val);
  }
}

template <typename K = uint64_t, typename V = float, typename S = uint64_t,
          int32_t TILE_SIZE = 32>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_1024) inline void assign_values_kernel_with_io_vf(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    uint64_t n, uint32_t thread_all, uint32_t block_index,
    uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  if (!buckets) {
    return;
  }
  if (!keys) {
    return;
  }
  if (!values) {
    return;
  }
  auto lane_id = threadIdx.x % TILE_SIZE;
  const uint64_t N = n * TILE_SIZE;

  for (uint64_t t = block_index * blockDim.x + threadIdx.x; t < N;
       t += thread_all) {
    uint64_t key_idx = t / TILE_SIZE;
    K key = keys[key_idx];
    if (IS_RESERVED_KEY<K>(key)) continue;
    const K hashed_key = Murmur3HashDevice(key);
    uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                         capacity_divisor_shift, capacity);
    uint32_t key_pos = global_idx & (bucket_max_size - 1);
    uint64_t bkt_idx = global_idx >> max_bucket_shift;

    __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
    __gm__ K* bucket_keys = bucket->keys_;
    __gm__ V* bucket_vectors = bucket->vectors;

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_and_lock_for_update<K, S, TILE_SIZE>(
        bucket_keys, bucket_max_size, key, key_pos, lane_id);

    if (occupy_result == OccupyResult::REFUSED) continue;
    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(values + key_idx * dim,
                                bucket_vectors + key_pos * dim, dim, lane_id);
      asc_threadfence();
    }
    if (lane_id == 0) {
      (void)asc_atomic_exch(bucket_keys + key_pos, key);
    }
  }
}

template <class K, class V, class S>
__global__ __vector__ void assign_values_kernel(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, __gm__ K* keys, __gm__ void* values,
    uint64_t n, uint32_t value_size, uint32_t max_bucket_shift,
    uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift,
    uint64_t n_align_warp, int32_t group_size) {
  constexpr uint32_t thread_num = 512;
  const uint32_t thread_all = thread_num * GetBlockNum();

  DISPATCH_VALUE_SIZE(
      value_size,
      (asc_vf_call<assign_values_kernel_with_digest_vf<K, V, S, DTYPE>>(
          dim3{static_cast<uint32_t>(thread_num)}, buckets, capacity,
          bucket_max_size, dim, keys, values, n, thread_all, GetBlockIdx(),
          max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift,
          n_align_warp, group_size)));
}

template <class K, class V, class S>
__global__ __vector__ void assign_values_kernel_with_io(
    __gm__ Bucket<K, V, S>* buckets, uint64_t capacity,
    uint32_t bucket_max_size, uint32_t dim, __gm__ K* keys, __gm__ V* values,
    uint64_t n, uint32_t max_bucket_shift, uint64_t capacity_divisor_magic,
    uint64_t capacity_divisor_shift) {
  constexpr uint32_t thread_num = 1024;
  const uint32_t thread_all = thread_num * GetBlockNum();
  asc_vf_call<assign_values_kernel_with_io_vf<K, V, S>>(
      dim3{static_cast<uint32_t>(thread_num)}, buckets, capacity,
      bucket_max_size, dim, keys, values, n, thread_all, GetBlockIdx(),
      max_bucket_shift, capacity_divisor_magic, capacity_divisor_shift);
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_ASSIGN_VALUES_KERNEL_H_