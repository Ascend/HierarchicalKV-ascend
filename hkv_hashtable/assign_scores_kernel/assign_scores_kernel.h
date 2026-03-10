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

/* !
 * \file assign_scores_kernel.h
 * \brief assign_scores_kernel
 */

 #ifndef ASCENDC_ASSIGN_SCORES_KERNEL_H_
 #define ASCENDC_ASSIGN_SCORES_KERNEL_H_
 
 #include <kernel_operator.h>
 #include <cstdint>
 #include "../../include/score_functor.h"
 #include "../../include/types.h"
 #include "../../include/utils.h"
 
 namespace npu {
 namespace hkv {
 using namespace AscendC;
 template <typename K = uint64_t, typename V = float, typename S = uint64_t, int Strategy = -1>
 __simt_vf__ __aicore__
 LAUNCH_BOUND(THREAD_NUM_512) inline void assign_scores_kernel_vf(
     __gm__ void* buckets_gm, uint64_t capacity, uint32_t bucket_capacity,
     uint32_t dim, __gm__ void* keys_gm, __gm__ void* scores_gm, uint64_t n,
     uint64_t global_epoch, const uint64_t total_thread_num,
     uint64_t system_cycle, uint32_t block_id, uint32_t max_bucket_shift,
     uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
   uint64_t kv_idx = block_id * blockDim.x + threadIdx.x;
 
   if (kv_idx >= n) {
     return;
   }
   if (!buckets_gm) {
     return;
   }
   if (!keys_gm) {
     return;
   }
 
   using ScoreFunctor = ScoreFunctor<K, V, S, Strategy>;
 
   __gm__ Bucket<K, V, S>* __restrict__ buckets =
       (__gm__ Bucket<K, V, S>*)buckets_gm;
   __gm__ K* __restrict__ keys = (__gm__ K*)keys_gm;
   __gm__ S* __restrict__ scores = (__gm__ S*)scores_gm;
 
   __gm__ K* bucket_keys_ptr = buckets->keys_;
 
   for (; kv_idx < n; kv_idx += total_thread_num) {
     K key = keys[kv_idx];
     if (!IS_RESERVED_KEY<K>(key)) {
       // 计算哈希和桶位置
       const K hashed_key = Murmur3HashDevice(key);
       uint64_t global_idx = get_global_idx(hashed_key, capacity_divisor_magic,
                                            capacity_divisor_shift, capacity);
       uint32_t key_pos = global_idx & (bucket_capacity - 1);
       uint64_t bkt_idx = global_idx >> max_bucket_shift;
 
       __gm__ Bucket<K, V, S>* bucket = buckets + bkt_idx;
       bucket_keys_ptr = bucket->keys_;
 
       // 查找并更新逻辑（简化版）
       uint32_t offset = 0;
       for (; offset < bucket_capacity; offset++) {
         uint32_t current_pos = (key_pos + offset) & (bucket_capacity - 1);
         __gm__ K* current_key_ptr = bucket_keys_ptr + current_pos;
         K current_key = *current_key_ptr;
         if (current_key == static_cast<K>(EMPTY_KEY)) {
           break;
         }
         if (current_key == key) {
           K try_key =
               Simt::AtomicCas(current_key_ptr, current_key, static_cast<K>(LOCKED_KEY));
           // 抢占成功
           if (try_key == current_key) {
             ScoreFunctor::update_without_missed(
                 bucket_keys_ptr, bucket_capacity, current_pos, scores, kv_idx,
                 global_epoch, system_cycle);
             (void)Simt::AtomicExch(current_key_ptr, key);
             break;
           }
         }
       }
     }
   }
 }
 
 template <class K, class V, class S, int Strategy = -1>
 __global__ __vector__ void assign_scores_kernel(
      __gm__ void* buckets, uint64_t capacity, uint32_t bucket_capacity, uint32_t dim,
     __gm__ void* keys, __gm__ void* scores, uint64_t n, uint64_t global_epoch,
     uint32_t value_size, uint32_t max_bucket_shift,
     uint64_t capacity_divisor_magic, uint64_t capacity_divisor_shift) {
 
   uint64_t system_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
   const uint64_t total_thread_num = THREAD_NUM_512 * GetBlockNum();
 
   Simt::VF_CALL<assign_scores_kernel_vf<K, V, S, Strategy>>(
       Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets, capacity,
               bucket_capacity, dim, keys, scores, n, global_epoch,
               total_thread_num, system_cycle, GetBlockIdx(), max_bucket_shift,
               capacity_divisor_magic, capacity_divisor_shift);
 }
 }  // namespace hkv
 }  // namespace npu
 
 #endif  // ASCENDC_ASSIGN_SCORES_KERNEL_H_
