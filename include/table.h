/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <acl/acl.h>
#include "allocator.h"
#include "debug.h"
#include "utils.h"
#include "types.h"

namespace npu {
namespace hkv {

/**
 * @brief Interface for bucket memory address provider.
 * Used by create_table / initialize_buckets / double_capacity so that
 * bucket memory allocation and address calculation are handled by the
 * implementation (e.g. BucketMemoryPoolManager); table code only calls
 * ensure_buckets_for_range and get_bucket_address / get_bucket_memory_size.
 */
struct IBucketAddressProvider {
  virtual void ensure_buckets_for_range(size_t start, size_t end,
                                        size_t num_of_buckets_per_alloc,
                                        BaseAllocator* allocator) = 0;
  virtual uint8_t* get_bucket_address(size_t bucket_index) const = 0;
  virtual size_t get_bucket_memory_size() const = 0;
  virtual bool use_pool() const = 0;
  virtual ~IBucketAddressProvider() = default;
};

template <class P>
void realloc(P* ptr, size_t old_size, size_t new_size,
             BaseAllocator* allocator) {
  old_size = std::min(old_size, new_size);

  char* new_ptr = nullptr;
  allocator->alloc(MemoryType::Device, reinterpret_cast<void**>(&new_ptr),
                   new_size);
  if (*ptr != nullptr) {
    NPU_CHECK(aclrtMemcpy(new_ptr, new_size, *ptr, old_size,
                          ACL_MEMCPY_DEVICE_TO_DEVICE));
    allocator->free(MemoryType::Device, *ptr);
  }

  NPU_CHECK(aclrtMemset((new_ptr + old_size), (new_size - old_size), 0,
                        (new_size - old_size)));

  *ptr = reinterpret_cast<P>(new_ptr);
  return;
}

template <class P>
void realloc_host(P* ptr, size_t old_size, size_t new_size,
                  BaseAllocator* allocator) {
  // Truncate old_size to limit dowstream copy ops.
  old_size = std::min(old_size, new_size);

  // Alloc new buffer and copy at old data.
  char* new_ptr = nullptr;
  allocator->alloc(MemoryType::Host, (void**)&new_ptr, new_size);

  if (*ptr != nullptr) {
    std::memcpy(new_ptr, *ptr, old_size);
    allocator->free(MemoryType::Host, *ptr);
  }

  // Zero-fill remainder.
  std::memset(new_ptr + old_size, 0, new_size - old_size);

  // Switch to new pointer.
  *ptr = reinterpret_cast<P>(new_ptr);
  return;
}

/* uint64_t快除法：
 * r = x / y，如果y的值固定，则除法可以等效替换为如下公式
 * 预计算部分（本函数内）：
 * 1. shift = ceil(log2(y))
 * 2. magic = ceil(2 ^ (64 + shift) / y)
 * 运行时计算部分（kernel内）：
 * 3. q = (x * magic) >> 64 由__umul64hi完成
 * 4. t = ((x - q) >> 1) + q
 * 5. r = t >> (shift - 1)
 */
template <class K, class V, class S>
void precomputation_for_kernel_div(Table<K, V, S>& table) {
  // x / max_bucket_size => x >> max_bucket_shift
  table.max_bucket_shift =
      static_cast<uint32_t>(std::log2(table.bucket_max_size));

#ifndef FORBID_QUICK_DIV
  unsigned __int128 one_u128 = 1;
  uint64_t divisor_shift = 0;
  const uint64_t& divisor = table.capacity;
  for (divisor_shift = 0; divisor_shift < 64; divisor_shift++) {
    if ((one_u128 << divisor_shift) >= divisor) {
      break;
    }
  }
  unsigned __int128 magic_u128 =
      ((one_u128 << 64) * ((one_u128 << divisor_shift) - divisor)) / divisor +
      1;
  table.capacity_divisor_magic = static_cast<uint64_t>(magic_u128);
  table.capacity_divisor_shift = divisor_shift - 1;
#endif
}

/* Initialize the buckets with index from start to end. */
template <class K, class V, class S>
void initialize_buckets(Table<K, V, S>** table, BaseAllocator* allocator,
                        const size_t start, const size_t end,
                        const uint32_t block_dim,
                        IBucketAddressProvider* provider = nullptr) {
  /* As testing results show us, when the number of buckets is greater than
   * the 4 million the performance will drop significantly, we believe the
   * to many pinned memory allocation causes this issue, so we change the
   * strategy to allocate some memory slices whose size is not greater than
   * 64GB, and put the buckets pointer point to the slices.
   */
  HKV_CHECK(start < end,
               "initialize_buckets, start should be less than end!");
  size_t buckets_num = end - start;
  const size_t total_size_of_vectors =
      buckets_num * (*table)->bucket_max_size * sizeof(V) * (*table)->dim;
  const size_t num_of_memory_slices =
      1 + (total_size_of_vectors - 1) / (*table)->bytes_per_slice;
  size_t num_of_buckets_in_one_slice =
      (*table)->bytes_per_slice /
      ((*table)->bucket_max_size * sizeof(V) * (*table)->dim);
  size_t num_of_allocated_buckets = 0;
  constexpr uint32_t value_size = sizeof(V);
  realloc_host<V**>(
      &((*table)->slices), (*table)->num_of_memory_slices * sizeof(V*),
      ((*table)->num_of_memory_slices + num_of_memory_slices) * sizeof(V*),
      allocator);

  bool mixed_hbm = false;
  for (size_t i = (*table)->num_of_memory_slices;
       i < (*table)->num_of_memory_slices + num_of_memory_slices; i++) {
    if (i == (*table)->num_of_memory_slices + num_of_memory_slices - 1) {
      num_of_buckets_in_one_slice = buckets_num - num_of_allocated_buckets;
    }
    size_t slice_real_size = num_of_buckets_in_one_slice *
                             (*table)->bucket_max_size * sizeof(V) *
                             (*table)->dim;
    if ((*table)->remaining_hbm_for_vectors >= slice_real_size) {
      if (!(*table)->is_pure_hbm) {
        mixed_hbm = true;
      }
      allocator->alloc(MemoryType::Device, (void**)&((*table)->slices[i]),
                       slice_real_size);
      (*table)->remaining_hbm_for_vectors -= slice_real_size;
    } else {
      HKV_CHECK(false,
                "Unsupport using host memory yet, please set "
                "max_hbm_for_vectors to a sufficiently large value.");
      (*table)->is_pure_hbm = false;
      allocator->alloc(MemoryType::Pinned, (void**)&((*table)->slices[i]),
                       slice_real_size);
    }
    for (size_t j = 0; j < num_of_buckets_in_one_slice; j++) {
      if ((*table)->is_pure_hbm || mixed_hbm) {
        size_t index = start + num_of_allocated_buckets + j;
        V* address =
            (*table)->slices[i] + j * (*table)->bucket_max_size * (*table)->dim;
        ACLRT_LAUNCH_KERNEL(allocate_bucket_vectors_kernel)(1, 0, (*table)->buckets, index, address, value_size);
      } else {
        V* h_ptr =
            (*table)->slices[i] + j * (*table)->bucket_max_size * (*table)->dim;
        V* address = nullptr;
        NPU_CHECK(aclrtHostRegister(h_ptr, slice_real_size, ACL_HOST_REGISTER_MAPPED, (void**)&address));
        size_t index = start + num_of_allocated_buckets + j;
        ACLRT_LAUNCH_KERNEL(allocate_bucket_vectors_kernel)(1, 0, (*table)->buckets, index, address, value_size);
      }
    }
    NPU_CHECK(aclrtSynchronizeDevice());
    num_of_allocated_buckets += num_of_buckets_in_one_slice;
  }

  (*table)->num_of_memory_slices += num_of_memory_slices;
  uint32_t bucket_max_size = static_cast<uint32_t>((*table)->bucket_max_size);
  size_t local_bucket_memory_size = bucket_max_size * (sizeof(K) + sizeof(S));
  // Align to the cache line size.
  constexpr uint32_t CACHE_LINE_SIZE = 128U / sizeof(uint8_t);
  uint32_t reserve_size =
      bucket_max_size < CACHE_LINE_SIZE ? CACHE_LINE_SIZE : bucket_max_size;
  local_bucket_memory_size += reserve_size * sizeof(uint8_t);

  size_t actual_bucket_memory_size = local_bucket_memory_size;
  if (provider != nullptr) {
    provider->ensure_buckets_for_range(start, end,
                                       (*table)->num_of_buckets_per_alloc,
                                       allocator);
    actual_bucket_memory_size = provider->get_bucket_memory_size();
  }

  HKV_CHECK(start % (*table)->num_of_buckets_per_alloc == 0,
               "initialize_buckets, start must be times of "
               "num_of_buckets_per_alloc!");
  /* NOTICE: Only the buckets which index is the times of
   * `num_of_buckets_per_alloc` will allocate a real address, that provides the
   * callers a method to avoid memory fragmentation.
   */
  for (size_t i = start; i < end; i += (*table)->num_of_buckets_per_alloc) {
    uint8_t* address = nullptr;
    size_t num_of_buckets =
        std::min(end - i, (*table)->num_of_buckets_per_alloc);

    if (provider != nullptr) {
      address = provider->get_bucket_address(i);
    } else {
      // Backward compatibility: use allocator when no provider
      allocator->alloc(MemoryType::Device, (void**)&(address),
                       actual_bucket_memory_size * num_of_buckets);
    }

    ACLRT_LAUNCH_KERNEL(allocate_bucket_others_kernel)(1, 0, (*table)->buckets, actual_bucket_memory_size, num_of_buckets, i,
                        address, reserve_size, bucket_max_size, value_size);
  }
  NPU_CHECK(aclrtSynchronizeDevice());

  {
    const size_t block_size = 512;
    const size_t N = end - start + 1;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    ACLRT_LAUNCH_KERNEL(create_atomic_keys_kernel)
        (grid_size, 0, (*table)->buckets, start, end, (*table)->bucket_max_size, value_size);
  }

  {
    ACLRT_LAUNCH_KERNEL(create_atomic_scores_kernel)
    (block_dim, 0, (*table)->buckets, start, end, (*table)->bucket_max_size, value_size);
  }
  NPU_CHECK(aclrtSynchronizeDevice());
  NpuCheckError();
}

template <class K, class V, class S>
size_t get_slice_size(Table<K, V, S>** table) {
  const size_t min_slice_size =
      (*table)->bucket_max_size * sizeof(V) * (*table)->dim;
  const size_t max_table_size = (*table)->max_size * sizeof(V) * (*table)->dim;
  size_t slice_size = 0;

  if (max_table_size >= GB(128)) {
    slice_size = GB(16);
  } else if (max_table_size >= GB(16)) {
    slice_size = GB(2);
  } else if (max_table_size >= GB(2)) {
    slice_size = MB(128);
  } else if (max_table_size >= MB(128)) {
    slice_size = MB(16);
  } else if (max_table_size >= MB(16)) {
    slice_size = MB(1);
  } else {
    slice_size = min_slice_size;
  }

  return std::max(min_slice_size, slice_size);
}

/* Initialize a Table struct.

   K: The key type
   V: The value type which should be static array type and C++ class
      with customized construct is not supported.
   S: The score type, the score will be used to store the timestamp
      or occurrence frequency or any thing for eviction.
   DIM: Vector dimension.
*/
template <class K, class V, class S>
void create_table(Table<K, V, S>** table, BaseAllocator* allocator,
                  const uint32_t block_dim,
                  const size_t dim, const size_t init_size = 134217728,
                  const size_t max_size = std::numeric_limits<size_t>::max(),
                  const size_t max_hbm_for_vectors = 0,
                  const size_t bucket_max_size = 128,
                  const size_t num_of_buckets_per_alloc = 1,
                  const size_t tile_size = 32, const bool primary = true,
                  IBucketAddressProvider* provider = nullptr) {
  allocator->alloc(MemoryType::Host, (void**)table, sizeof(Table<K, V, S>));
  (void)std::memset((void*)*table, 0, sizeof(Table<K, V, S>));
  (*table)->dim = dim;
  (*table)->bucket_max_size = bucket_max_size;
  (*table)->max_size = std::max(init_size, max_size);
  (*table)->tile_size = tile_size;
  (*table)->is_pure_hbm = true;
  (*table)->bytes_per_slice = get_slice_size<K, V, S>(table);
  (*table)->num_of_buckets_per_alloc = num_of_buckets_per_alloc;

  // The bucket number will be the minimum needed for saving memory if no
  // rehash.
  if ((init_size * 2) > (*table)->max_size) {
    (*table)->buckets_num =
        1 + (((*table)->max_size - 1) / (*table)->bucket_max_size);
  } else {
    (*table)->buckets_num = 1;
    while ((*table)->buckets_num * (*table)->bucket_max_size < init_size) {
      (*table)->buckets_num *= 2;
    }
  }

  (*table)->capacity = (*table)->buckets_num * (*table)->bucket_max_size;
  (*table)->max_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->remaining_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->primary = primary;
  precomputation_for_kernel_div(**table);

  allocator->alloc(MemoryType::Device, (void**)&((*table)->buckets_size),
                   (*table)->buckets_num * sizeof(int));
  NPU_CHECK(aclrtMemset((*table)->buckets_size, (*table)->buckets_num * sizeof(int), 0,
                        (*table)->buckets_num * sizeof(int)));

  allocator->alloc(MemoryType::Device, (void**)&((*table)->buckets),
                   (*table)->buckets_num * sizeof(Bucket<K, V, S>));
  NPU_CHECK(aclrtMemset((*table)->buckets, (*table)->buckets_num * sizeof(Bucket<K, V, S>), 0,
                        (*table)->buckets_num * sizeof(Bucket<K, V, S>)));

  initialize_buckets<K, V, S>(table, allocator, 0, (*table)->buckets_num, block_dim,
                              provider);
  NpuCheckError();
}

/* free all of the resource of a Table. */
template <class K, class V, class S>
void destroy_table(Table<K, V, S>** table, BaseAllocator* allocator,
                   bool use_memory_pool = false) {
  // Bucket memory is always managed by IBucketAddressProvider (e.g. BucketMemoryPoolManager);
  // skip individual bucket freeing here in both pool and non-pool cases.
  (void)use_memory_pool;

  for (size_t i = 0; i < (*table)->num_of_memory_slices; i++) {
    if (is_on_device((*table)->slices[i])) {
      allocator->free(MemoryType::Device, (*table)->slices[i]);
    } else {
      allocator->free(MemoryType::Pinned, (*table)->slices[i]);
    }
  }

  allocator->free(MemoryType::Host, (*table)->slices);
  allocator->free(MemoryType::Device, (*table)->buckets_size);
  allocator->free(MemoryType::Device, (*table)->buckets);
  allocator->free(MemoryType::Host, *table);
  NPU_CHECK(aclrtSynchronizeDevice());
  NpuCheckError();
}

template <class K, class V, class S>
void double_capacity(Table<K, V, S>** table, BaseAllocator* allocator,
                     const uint32_t block_dim,
                     IBucketAddressProvider* provider = nullptr) {
  realloc(&((*table)->buckets_size), (*table)->buckets_num * sizeof(int),
          (*table)->buckets_num * sizeof(int) * 2, allocator);

  realloc(&((*table)->buckets), (*table)->buckets_num * sizeof(Bucket<V, V, S>),
          (*table)->buckets_num * sizeof(Bucket<K, V, S>) * 2, allocator);

  initialize_buckets(table, allocator, (*table)->buckets_num,
                     (*table)->buckets_num * 2, block_dim,
                     provider);

  (*table)->capacity *= 2;
  (*table)->buckets_num *= 2;

  precomputation_for_kernel_div(**table);
}
}  // namespace hkv
}  // namespace npu