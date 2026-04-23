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
 * \file find_or_insert_ptr_kernel.h
 * \brief find_or_insert_ptr_kernel
 */

 #ifndef ASCENDC_DUMP_KERNEL_H_
 #define ASCENDC_DUMP_KERNEL_H_
 
 #include <kernel_operator.h>
 #include <cstdint>
 
 #include "types.h"
 #include "utils.h"
 #include "simt_vf_dispatcher.h"
 #include "simt_api/asc_simt.h"
 
 namespace npu {
 namespace hkv {
 using namespace AscendC;
 
 template <class K, class V, class S, class VecV>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_2048) inline void dump_kernel_vf(
     __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets, __gm__ K* d_key, __gm__ V* d_val,
     __gm__ S* d_score, const size_t offset, const size_t search_length, const size_t search_length_align,
     const uint64_t total_thread_num, __gm__ size_t* d_dump_counter,
     uint64_t ub_shared_kvs_mem, __ubuf__ uint32_t* block_acc,
     __ubuf__ uint32_t* global_acc, int32_t group_size, uint32_t dim_in, uint32_t blockIdx) {
   if ((!table) || (!buckets) || (!d_key) || (!d_val) || (!d_dump_counter)) {
     return;
   }
 
   auto block_tuples =
       reinterpret_cast<__ubuf__ KVM<K, S>*>(ub_shared_kvs_mem);

   __gm__ VecV* __restrict__ d_vecv_val = reinterpret_cast<__gm__ VecV*>(d_val);
 
   const size_t bucket_max_size{table->bucket_max_size};
 
   size_t tid{blockIdx * blockDim.x + threadIdx.x};
   for (; tid < search_length_align; tid += total_thread_num) {
     if (threadIdx.x == 0) {
       block_acc[0] = 0;
     }
     asc_syncthreads();
     uint32_t choose_flag = 0;
     size_t local_index = 0;
     if (tid < search_length) {
       __gm__ Bucket<K, V, S>* bucket =
           buckets + (tid + offset) / bucket_max_size;
       __gm__ K* bucket_keys_ptr = bucket->keys_;
       __gm__ VecV* bucket_values_ptr =
           reinterpret_cast<__gm__ VecV*>(bucket->vectors);
       __gm__ S* bucket_scores_ptr = bucket->scores_;

       const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
       const K key{bucket_keys_ptr[key_idx]};

       if (!IS_RESERVED_KEY<K>(key)) {
         choose_flag = 1;
         local_index = atomicAdd(block_acc, 1u);
         block_tuples[local_index].key = key;
         block_tuples[local_index].value =
             reinterpret_cast<uint64_t>(&bucket_values_ptr[key_idx * dim_in]);
         block_tuples[local_index].score = bucket_scores_ptr[key_idx];
       }
       asc_syncthreads();

       if (threadIdx.x == 0) {
         global_acc[0] =
             atomicAdd(d_dump_counter, static_cast<size_t>(block_acc[0]));
       }
       asc_syncthreads();

       if (block_acc[0] == 0) {
         continue;
       }
     }

     auto cg_rank_id = threadIdx.x % group_size;
     auto cg_rank_id_start = threadIdx.x - cg_rank_id;
     uint64_t tuple_value_ptr = 0;
     __ubuf__ const KVM<K, S>& tuple{block_tuples[local_index]};
     const size_t j{global_acc[0] + local_index};
     if (choose_flag == 1) {
       d_key[j] = tuple.key;

       if (d_score != nullptr) {
         d_score[j] = tuple.score;
       }
     }

     for (int32_t i = 0; i < group_size; i++) {
       auto cg_flag = asc_shfl(choose_flag, i, group_size);
       if (cg_flag == 1) {
         // 协程组并行写入向量值
         auto val_start = asc_shfl(j, i, group_size);
         uint64_t value_sync_ptr = asc_shfl(tuple.value, i, group_size);
         auto value_sync = reinterpret_cast<__gm__ VecV*>(value_sync_ptr);
         for (uint32_t k = cg_rank_id; k < dim_in; k += group_size) {
           __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                 L1CacheType::NON_CACHEABLE>(d_vecv_val + val_start * dim_in + k,
                                             value_sync[k]);
         }
       }
     }
   }
 }

 template <class K, class V, class S>
 __global__ __vector__ void dump_kernel(
     __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
     __gm__ K* keys, __gm__ V* vals, __gm__ S* scores, const size_t offset,
     const size_t search_length, const size_t search_length_align,
     __gm__ size_t* dump_counter, uint32_t value_size, int32_t group_size,
     uint32_t dim) {
   const uint64_t total_thread_num = THREAD_NUM_2048 * GetBlockNum();
 
   AscendC::TPipe pipe;
 
   AscendC::TBuf<AscendC::TPosition::VECCALC> shared_kvs_mem;
   pipe.InitBuffer(shared_kvs_mem, THREAD_NUM_2048 * sizeof(KVM<K, S>));
   AscendC::LocalTensor<KVM<K, S>> shared_kvs_tensor =
       shared_kvs_mem.Get<KVM<K, S>>();
   AscendC::TBuf<AscendC::TPosition::VECCALC> block_acc;
   pipe.InitBuffer(block_acc, sizeof(uint32_t));
   AscendC::LocalTensor<uint32_t> shared_block_acc_tensor =
       block_acc.Get<uint32_t>();
   __ubuf__ uint32_t* ub_shared_block_acc_mem =
       reinterpret_cast<__ubuf__ uint32_t*>(
           shared_block_acc_tensor.GetPhyAddr());
 
   AscendC::TBuf<AscendC::TPosition::VECCALC> global_acc;
   pipe.InitBuffer(global_acc, sizeof(uint32_t));
   AscendC::LocalTensor<uint32_t> shared_global_acc_tensor =
       global_acc.Get<uint32_t>();
   __ubuf__ uint32_t* ub_shared_global_acc_mem =
       reinterpret_cast<__ubuf__ uint32_t*>(
           shared_global_acc_tensor.GetPhyAddr());
 
   DISPATCH_VALUE_SIZE(
         value_size,
         (Simt::VF_CALL<dump_kernel_vf<K, V, S, DTYPE>>(
         Simt::Dim3{static_cast<uint32_t>(THREAD_NUM_2048)}, table, buckets,
               keys, vals, scores, offset, search_length, search_length_align, total_thread_num,
               dump_counter, shared_kvs_tensor.GetPhyAddr(),
               ub_shared_block_acc_mem, ub_shared_global_acc_mem, group_size, dim,
               GetBlockIdx())));
 }

 template <class K, class V, class S>
 __simt_vf__ __aicore__
 LAUNCH_BOUND(THREAD_NUM_2048) inline void dump_kernel_hybrid_vf(
     __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
     __gm__ K* d_key, __gm__ V * __gm__ * dst_val, __gm__ S* d_score,
     const size_t offset, const size_t search_length,
     const uint64_t total_thread_num, __gm__ size_t* d_dump_counter,
     __ubuf__ uint32_t* block_acc, __ubuf__ uint32_t* global_acc,
     uint32_t dim_in, uint32_t blockIdx) {
   if ((!table) || (!buckets) || (!d_key) || (!dst_val) || (!d_dump_counter)) {
     return;
   }
 
   const size_t bucket_max_size{table->bucket_max_size};
   size_t tid{blockIdx * blockDim.x + threadIdx.x};
   for (; tid < search_length; tid += total_thread_num) {
     if (threadIdx.x == 0) {
       block_acc[0] = 0;
     }
     asc_syncthreads();
     uint32_t choose_flag = 0;
     size_t local_index = 0;
     __gm__ Bucket<K, V, S>* bucket =
         buckets + (tid + offset) / bucket_max_size;
     __gm__ K* bucket_keys_ptr = bucket->keys_;
     __gm__ V* bucket_values_ptr = bucket->vectors;
     __gm__ S* bucket_scores_ptr = bucket->scores_;

     const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
     const K key{bucket_keys_ptr[key_idx]};

     if (!IS_RESERVED_KEY<K>(key)) {
       choose_flag = 1;
       local_index = atomicAdd(block_acc, 1u);
     }
     asc_syncthreads();

     if (threadIdx.x == 0) {
       global_acc[0] =
           atomicAdd(d_dump_counter, static_cast<size_t>(block_acc[0]));
     }
     asc_syncthreads();

     if (choose_flag == 1) {
       const size_t j{global_acc[0] + local_index};
       d_key[j] = key;
       dst_val[j] = (&bucket_values_ptr[key_idx * dim_in]);

       if (d_score != nullptr) {
         d_score[j] = bucket_scores_ptr[key_idx];
       }
     }

     if (block_acc[0] == 0) {
       continue;
     }
   }
 }

 template <class K, class V, class S>
 __global__ __vector__ void dump_kernel_hybrid(
     __gm__ Table<K, V, S>* table, __gm__ Bucket<K, V, S>* buckets,
     __gm__ K* keys, __gm__ V * __gm__ * src_vals, __gm__ S* scores,
     const size_t offset, const size_t search_length,
     __gm__ size_t* dump_counter, uint32_t dim) {
   const uint64_t total_thread_num = THREAD_NUM_2048 * GetBlockNum();
 
   AscendC::TPipe pipe;

   AscendC::TBuf<AscendC::TPosition::VECCALC> block_acc;
   pipe.InitBuffer(block_acc, sizeof(uint32_t));
   AscendC::LocalTensor<uint32_t> shared_block_acc_tensor =
       block_acc.Get<uint32_t>();
   __ubuf__ uint32_t* ub_shared_block_acc_mem =
       reinterpret_cast<__ubuf__ uint32_t*>(
           shared_block_acc_tensor.GetPhyAddr());
 
   AscendC::TBuf<AscendC::TPosition::VECCALC> global_acc;
   pipe.InitBuffer(global_acc, sizeof(uint32_t));
   AscendC::LocalTensor<uint32_t> shared_global_acc_tensor =
       global_acc.Get<uint32_t>();
   __ubuf__ uint32_t* ub_shared_global_acc_mem =
       reinterpret_cast<__ubuf__ uint32_t*>(
           shared_global_acc_tensor.GetPhyAddr());

   asc_vf_call<dump_kernel_hybrid_vf<K, V, S>>(
       dim3{static_cast<uint32_t>(THREAD_NUM_2048)}, table, buckets, keys,
       src_vals, scores, offset, search_length, total_thread_num, dump_counter,
       ub_shared_block_acc_mem, ub_shared_global_acc_mem, dim, GetBlockIdx());
 }

 template <class V>
 __global__ __vector__ void export_batch_value_kernel(
     uint32_t block_dim, uint32_t dim, uint32_t max_tile_size, __gm__ V* values,
     __gm__ size_t* dump_counter, __gm__ V * __gm__ * d_src_values) {
   size_t n = *dump_counter;
   uint64_t tail_core_move_num = n / block_dim;
   uint64_t former_core_move_num = tail_core_move_num + 1;
   uint32_t former_num = n - tail_core_move_num * block_dim;
   uint32_t tile_size = (dim <= max_tile_size) ? dim : max_tile_size;
   uint32_t num_tiles = (dim + tile_size - 1) / tile_size;

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

   TPipe pipe;
   TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> move_queue;
   pipe.InitBuffer(move_queue, DOUBLE_BUFFER, tile_size * sizeof(V));

   GlobalTensor<V> src_values_gm;
   GlobalTensor<V> dst_values_gm;
   LocalTensor<V> move_local;
   DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};

   dst_values_gm.SetGlobalBuffer(values);

   for (uint64_t i = core_start_idx; i < core_start_idx + core_move_count;
        i++) {
     __gm__ V* src_value = d_src_values[i];
     if (src_value == nullptr) {
       continue;
     }

     uint64_t dst_offset = i * dim;

     for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
       uint32_t current_tile_size = (tile_idx == num_tiles - 1)
                                        ? (dim - tile_idx * tile_size)
                                        : tile_size;
       DataCopyExtParams copy_params{
           1, static_cast<uint32_t>(current_tile_size * sizeof(V)), 0, 0, 0};

       move_queue.AllocTensor<V>(move_local);

       src_values_gm.SetGlobalBuffer(src_value + tile_idx * tile_size);
       DataCopyPad(move_local, src_values_gm, copy_params, pad_params);
       move_queue.EnQue<V>(move_local);
       move_queue.DeQue<V>(move_local);

       DataCopyPad(dst_values_gm[dst_offset + tile_idx * tile_size], move_local,
                   copy_params);

       move_queue.FreeTensor(move_local);
     }
   }
 }

 }  // namespace hkv
 }  // namespace npu
 
 #endif  // ASCENDC_DUMP_KERNEL_H_
 