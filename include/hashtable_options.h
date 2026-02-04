/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include "memory_pool.h"

namespace npu {
namespace hkv {

/**
 * @brief The options struct of HierarchicalKV.
 *
 * @note This header is for host-only code. Do not include it from NPU kernel
 * sources (types.h is included by kernel code and must not pull in memory_pool
 * or acl).
 */
struct HashTableOptions {
  size_t init_capacity = 0;        ///< The initial capacity of the hash table.
  size_t max_capacity = 0;         ///< The maximum capacity of the hash table.
  size_t max_hbm_for_vectors = 0;  ///< The maximum HBM for vectors, in bytes.
  size_t max_bucket_size = 128;    ///< The length of each bucket.
  size_t dim = 64;                 ///< The dimension of the vectors.
  float max_load_factor = 0.5f;    ///< The max load factor before rehashing.
  int block_size = 128;            ///< The default block size for CANN kernels.
  int io_block_size = 1024;        ///< The block size for IO CANN kernels.
  int device_id = -1;              ///< The ID of device.
  bool io_by_cpu = false;  ///< The flag indicating if the CPU handles IO.
  bool use_constant_memory = false;  ///< reserved
  /*
   * reserved_key_start_bit = 0, is the default behavior, HKV reserves
   * `0xFFFFFFFFFFFFFFFD`, `0xFFFFFFFFFFFFFFFE`, and `0xFFFFFFFFFFFFFFFF`  for
   * internal using. if the default one conflicted with your keys, change the
   * reserved_key_start_bit value to a numbers between 1 and 62,
   * reserved_key_start_bit = 1 means using the insignificant bits index 1 and 2
   * as the keys as the reserved keys and the index 0 bit is 0 and all the other
   * bits are 1, the new reserved keys are `FFFFFFFFFFFFFFFE`,
   * `0xFFFFFFFFFFFFFFFC`, `0xFFFFFFFFFFFFFFF8`, and `0xFFFFFFFFFFFFFFFA` the
   * console log prints the reserved keys during the table initialization.
   */
  int reserved_key_start_bit = 0;       ///< The binary index of reserved key.
  size_t num_of_buckets_per_alloc = 1;  ///< Number of buckets allocated in each
                                        ///< HBM allocation, must be power of 2.
  bool api_lock = true;  ///<  The flag indicating whether to lock the table
                         ///<  once enters the API.
  MemoryPoolOptions
      device_memory_pool;  ///< Configuration options for device memory pool.
  MemoryPoolOptions
      host_memory_pool;  ///< Configuration options for host memory pool.
};

}  // namespace hkv
}  // namespace npu
