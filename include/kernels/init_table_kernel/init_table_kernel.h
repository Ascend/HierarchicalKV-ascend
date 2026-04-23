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
 * \file init_table_kernel.h
 * \brief init_table_kernel
 */

 #ifndef ASCENDC_INIT_TABLE_KERNEL_H_
 #define ASCENDC_INIT_TABLE_KERNEL_H_
 
 #include <kernel_operator.h>
 #include <cstddef>
 #include <cstdint>
 #include "types.h"
 #include "utils.h"
 
 namespace npu {
 namespace hkv {
 using namespace AscendC;
 
 template <class K, class V, class S>
 __simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void allocate_bucket_vectors_vf(
     __gm__ Bucket<K, V, S>* buckets_gm, const size_t index, __gm__ V* address_gm) {
   __gm__ Bucket<K, V, S>* buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
   __gm__ V* address = (__gm__ V*)address_gm;
   buckets[index].vectors = address;
 }
 
 template <class K, class V, class S>
 __simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void allocate_bucket_others_vf(
     __gm__ Bucket<K, V, S>* buckets_gm, size_t total_size_per_bucket, size_t num_of_buckets,
     const int start_index, __gm__ uint8_t* address, const uint32_t reserve_size,
     const size_t bucket_max_size) {
   __gm__ Bucket<K, V, S>* buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
   for (size_t step = 0; step < num_of_buckets; step++) {
     size_t index = start_index + step;
     buckets[index].digests_ = reinterpret_cast<__gm__ uint8_t*>(address);
     buckets[index].keys_ =
         reinterpret_cast<__gm__ K*>(buckets[index].digests_ + reserve_size);
     buckets[index].scores_ =
         reinterpret_cast<__gm__ S*>(buckets[index].keys_ + bucket_max_size);
     address += total_size_per_bucket;
   }
 }
 
 template <class K, class V, class S>
 __simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void get_bucket_others_address_vf(
     __gm__ Bucket<K, V, S>* buckets_gm, const int index, __gm__ uint8_t*__gm__* address_gm) {
   __gm__ Bucket<K, V, S>* buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
   __gm__ uint8_t*__gm__* address = reinterpret_cast<__gm__ uint8_t*__gm__* >(address_gm);
 
   *address = reinterpret_cast<__gm__ uint8_t*>(buckets[index].digests_);
 }
 
 template <class K, class V, class S>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_512) inline void create_atomic_keys_vf(
     __gm__ Bucket<K, V, S>* buckets_gm, const size_t start,
     const size_t end, const size_t bucket_max_size, uint32_t blockIdx) {
   __gm__ Bucket<K, V, S>* buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
 
   size_t tid = (blockIdx * blockDim.x) + threadIdx.x;
   if (start + tid < end) {
     for (size_t i = 0; i < bucket_max_size; i++) {
       buckets[start + tid].digests_[i] = empty_digest<K>();
       buckets[start + tid].keys_[i] = static_cast<K>(EMPTY_KEY);
     }
   }
 }
 
 template <class K, class V, class S>
 __simt_vf__ __aicore__
 LAUNCH_BOUND(THREAD_NUM_2048) inline void create_atomic_scores_vf(
     __gm__ Bucket<K, V, S>* buckets_gm, const size_t start, const size_t end,
     const size_t bucket_max_size, uint32_t blockIdx, uint64_t thread_all) {
   __gm__ Bucket<K, V, S>* buckets = reinterpret_cast<__gm__ Bucket<K, V, S>* __restrict>(buckets_gm);
 
   for (size_t tid = (block_idx * blockDim.x) + threadIdx.x; start + tid < end;
        tid += thread_all) {
     for (size_t i = 0; i < bucket_max_size; i++) {
       buckets[start + tid].scores_[i] = static_cast<S>(EMPTY_SCORE);
     }
   }
 }
 
 template <class K, class V, class S>
 __global__ __vector__ void allocate_bucket_others_kernel(
     __gm__ Bucket<K, V, S>* buckets_gm, size_t total_size_per_bucket, size_t num_of_buckets,
     const int start_index, __gm__ uint8_t* address, const uint32_t reserve_size,
     const size_t bucket_max_size) {
   Simt::VF_CALL<allocate_bucket_others_vf<K, V, S>>(
           Simt::Dim3{static_cast<uint32_t>(1)}, buckets_gm,
           total_size_per_bucket, num_of_buckets, start_index, address,
           reserve_size, bucket_max_size);
 }
 
 template <class K, class V, class S>
 __global__ __vector__ void create_atomic_keys_kernel(
     __gm__ Bucket<K, V, S>* buckets_gm, const size_t start, const size_t end,
     const size_t bucket_max_size) {
   Simt::VF_CALL<create_atomic_keys_vf<K, V, S>>(
       Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_512)}, buckets_gm, start,
                       end, bucket_max_size, GetBlockIdx());
 }
 
 template <class K, class V, class S>
 __global__ __vector__ void create_atomic_scores_kernel(
     __gm__ Bucket<K, V, S>* buckets_gm, const size_t start, const size_t end,
     const size_t bucket_max_size) {
 
   const uint64_t thread_all = THREAD_NUM_2048 * GetBlockNum();
   Simt::VF_CALL<create_atomic_scores_vf<K, V, S>>(
           Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_2048)}, buckets_gm, start,
           end, bucket_max_size, GetBlockIdx(), thread_all);
 }
 
 
 template <class K, class V, class S>
 __global__ __vector__ void get_bucket_others_address_kernel(
     __gm__ Bucket<K, V, S>* buckets_gm, const int index, __gm__ void* address_gm) {
 
   Simt::VF_CALL<get_bucket_others_address_vf<K, V, S>>(
           Simt::Dim3{static_cast<uint32_t>(1)}, buckets_gm, index, address_gm);
 }
 
 template <class K, class V, class S>
 __global__ __vector__ void allocate_bucket_vectors_kernel(
     __gm__ Bucket<K, V, S>* buckets_gm, const size_t index,  __gm__ V* address_gm) {
   Simt::VF_CALL<allocate_bucket_vectors_vf<K, V, S>>(
           Simt::Dim3{static_cast<uint32_t>(1)}, buckets_gm, index,
           address_gm);
 }
 }  // namespace hkv
 }  // namespace npu
 
 #endif  // ASCENDC_INIT_TABLE_KERNEL_H_
 