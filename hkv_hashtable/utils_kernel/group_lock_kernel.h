/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef ASCENDC_GROUP_LOCK_KERNEL_H_
#define ASCENDC_GROUP_LOCK_KERNEL_H_

#include <kernel_operator.h>
#include <simt_api/common_functions.h>
#include <cstdint>
#include "simt_api/device_atomic_functions.h"

namespace npu {
namespace hkv {
namespace group_lock {

/**
 * @brief Initialize the group lock counters on device
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void init_kernel_vf(
    __gm__ int32_t* update_count, __gm__ int32_t* read_count,
    __gm__ int32_t* unique_flag) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        update_count, 0);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        read_count, 0);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        unique_flag, 0);
  }
}
template <typename T>
__global__ __vector__ void init_kernel(__gm__ int32_t* d_update_count,
                                       __gm__ int32_t* d_read_count,
                                       __gm__ int32_t* d_unique_flag) {
  asc_vf_call<init_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)}, d_update_count,
                                 d_read_count, d_unique_flag);
}

/**
 * @brief Lock read kernel - marks that a read lock is being acquired
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void lock_read_kernel_vf(
    __gm__ int32_t* update_count, __gm__ int32_t* read_count) {
  for (;;) {
    int32_t tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                        L1CacheType::NON_CACHEABLE>(update_count);
    while (tmp) {
      tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(update_count);
    }
    asc_atomic_add(read_count, 1);
    tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(update_count);
    if (tmp == 0) {
      break;
    }
    asc_atomic_sub(read_count, 1);
  }
}

template <typename T>
__global__ __vector__ void lock_read_kernel(__gm__ int32_t* d_update_count,
                                            __gm__ int32_t* d_read_count) {
  asc_vf_call<lock_read_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                      d_update_count, d_read_count);
}

/**
 * @brief Unlock read kernel
 */
template <typename T>
__simt_vf__ __aicore__
LAUNCH_BOUND(1) inline void unlock_read_kernel_vf(__gm__ int32_t* read_count) {
  asc_atomic_sub(read_count, 1);
}

template <typename T>
__global__ __vector__ void unlock_read_kernel(__gm__ int32_t* d_read_count) {
  asc_vf_call<unlock_read_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                        d_read_count);
}

/**
 * @brief Lock update kernel
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void lock_update_kernel_vf(
    __gm__ int32_t* update_count, __gm__ int32_t* read_count) {
  for (;;) {
    int32_t tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                        L1CacheType::NON_CACHEABLE>(read_count);
    while (tmp) {
      tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(read_count);
    }
    asc_atomic_add(update_count, 1);
    tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(read_count);
    if (tmp == 0) {
      break;
    }
    asc_atomic_sub(update_count, 1);
  }
}

template <typename T>
__global__ __vector__ void lock_update_kernel(__gm__ int32_t* d_update_count,
                                              __gm__ int32_t* d_read_count) {
  asc_vf_call<lock_update_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                        d_update_count, d_read_count);
}

/**
 * @brief Unlock update kernel
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void unlock_update_kernel_vf(
    __gm__ int32_t* update_count) {
  asc_atomic_sub(update_count, 1);
}
template <typename T>
__global__ __vector__ void unlock_update_kernel(
    __gm__ int32_t* d_update_count) {
  asc_vf_call<unlock_update_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                          d_update_count);
}

/**
 * @brief Lock update_read (exclusive) kernel
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void lock_update_read_kernel_vf(
    __gm__ int32_t* update_count, __gm__ int32_t* read_count,
    __gm__ int32_t* unique_flag) {
  /* Lock unique flag */
  int32_t expected = 0;
  // 如果unique_flag==expected，当前线程获得锁，跳出while循环，否则表示其他线程已经获得锁，需要自旋直到拿到锁
  while (asc_atomic_cas(unique_flag, expected, 1) != expected) {
    expected = 0;
  }

  /* Ban update */
  for (;;) {
    int32_t tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                        L1CacheType::NON_CACHEABLE>(update_count);
    while (tmp) {
      tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(update_count);
    }
    asc_atomic_add(read_count, 1);
    tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(update_count);
    if (tmp == 0) {
      break;
    }
    asc_atomic_sub(read_count, 1);
  }
  /* Ban read */
  for (;;) {
    int32_t tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                        L1CacheType::NON_CACHEABLE>(read_count);
    while (tmp > 1) {
      tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                  L1CacheType::NON_CACHEABLE>(read_count);
    }
    asc_atomic_add(update_count, 1);
    tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                L1CacheType::NON_CACHEABLE>(read_count);
    if (tmp == 1) {
      break;
    }
    asc_atomic_sub(update_count, 1);
  }
}
template <typename T>
__global__ __vector__ void lock_update_read_kernel(
    __gm__ int32_t* d_update_count, __gm__ int32_t* d_read_count,
    __gm__ int32_t* d_unique_flag) {
  asc_vf_call<lock_update_read_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                             d_update_count, d_read_count,
                                             d_unique_flag);
}

/**
 * @brief Unlock update_read kernel
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void unlock_update_read_kernel_vf(
    __gm__ int32_t* update_count, __gm__ int32_t* read_count,
    __gm__ int32_t* unique_flag) {
  asc_atomic_sub(read_count, 1);
  asc_atomic_sub(update_count, 1);
  __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
      unique_flag, 0);
}
template <typename T>
__global__ __vector__ void unlock_update_read_kernel(
    __gm__ int32_t* d_update_count, __gm__ int32_t* d_read_count,
    __gm__ int32_t* d_unique_flag) {
  asc_vf_call<unlock_update_read_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                               d_update_count, d_read_count,
                                               d_unique_flag);
}

/**
 * @brief Get update count kernel
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void update_count_kernel_vf(
    __gm__ int32_t* counter, __gm__ int32_t* update_count) {
  auto tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN,
                   L1CacheType::CACHEABLE>(update_count);
  __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
      counter, tmp);
}

template <typename T>
__global__ __vector__ void update_count_kernel(__gm__ int32_t* d_counter,
                                               __gm__ int32_t* d_update_count) {
  asc_vf_call<update_count_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                         d_counter, d_update_count);
}

/**
 * @brief Get read count kernel
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void read_count_kernel_vf(
    __gm__ int32_t* counter, __gm__ int32_t* read_count) {
  auto tmp = __ldg<LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN,
                   L1CacheType::CACHEABLE>(read_count);
  __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
      counter, tmp);
}
template <typename T>
__global__ __vector__ void read_count_kernel(__gm__ int32_t* d_counter,
                                             __gm__ int32_t* d_read_count) {
  asc_vf_call<read_count_kernel_vf<T>>(dim3{static_cast<uint32_t>(1)},
                                       d_counter, d_read_count);
}

}  // namespace group_lock
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_GROUP_LOCK_KERNEL_H_
