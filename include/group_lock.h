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

#pragma once

#include <acl/acl.h>
#include <atomic>
#include <cassert>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include "../hkv_hashtable/utils_kernel/group_lock_kernel.h"
#include "debug.h"

namespace npu {
namespace hkv {
/**
 * @brief ACL-based group shared mutex implementation
 *
 * Implements a triple-group, mutex and relative lock guard for better E2E
 * performance:
 * - There are three roles: `inserter`, `updater`, and `reader`.
 * - Allow only one inserter to be executed concurrently.
 * - Allow multiple updaters to be executed concurrently.
 * - Allow multiple readers to be executed concurrently.
 * - Not allow inserter, readers and updaters to run concurrently
 * - The `update_read_lock` is exclusive and used for special APIs (like
 * `reserve`, `erase`, `clear`, etc.)
 */
class group_shared_mutex {
 public:
  group_shared_mutex(const group_shared_mutex&) = delete;
  group_shared_mutex& operator=(const group_shared_mutex&) = delete;

  group_shared_mutex() noexcept
      : h_update_count_(0), h_read_count_(0), h_unique_flag_(false) {
    // Allocate device memory for counters
    NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_update_count_),
                          sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_read_count_),
                          sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(&d_unique_flag_),
                          sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    // Initialize device counters
    group_lock::init_kernel<int32_t>
        <<<1, 0, 0>>>(d_update_count_, d_read_count_, d_unique_flag_);
    NPU_CHECK(aclrtSynchronizeDevice());
  }

  ~group_shared_mutex() noexcept {
    NPU_CHECK(aclrtSynchronizeDevice());
    if (d_update_count_) {
      aclrtFree(d_update_count_);
    }
    if (d_read_count_) {
      aclrtFree(d_read_count_);
    }
    if (d_unique_flag_) {
      aclrtFree(d_unique_flag_);
    }
  }

  /**
   * @brief Acquire shared read lock
   */
  void lock_read() {
    for (;;) {
      // Spin while there are updaters
      while (h_update_count_.load(std::memory_order_acquire)) {
      }

      // Try to increment read count
      h_read_count_.fetch_add(1, std::memory_order_acq_rel);

      // Check if no updaters now
      if (h_update_count_.load(std::memory_order_acquire) == 0) {
        // Sync with device
        aclrtStream stream;
        NPU_CHECK(aclrtCreateStream(&stream));
        group_lock::lock_read_kernel<int32_t>
            <<<1, 0, stream>>>(d_update_count_, d_read_count_);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        NPU_CHECK(aclrtDestroyStream(stream));
        break;
      }

      // Failed to get lock, decrement and retry
      h_read_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  /**
   * @brief Release shared read lock
   */
  void unlock_read(aclrtStream stream) {
    group_lock::unlock_read_kernel<int32_t><<<1, 0, stream>>>(d_read_count_);
    h_read_count_.fetch_sub(1, std::memory_order_release);
  }

  /**
   * @brief Acquire shared update lock
   */
  void lock_update() {
    for (;;) {
      // Spin while there are readers
      while (h_read_count_.load(std::memory_order_acquire)) {
      }

      // Try to increment update count
      h_update_count_.fetch_add(1, std::memory_order_acq_rel);

      // Check if no readers now
      if (h_read_count_.load(std::memory_order_acquire) == 0) {
        // Sync with device
        aclrtStream stream;
        NPU_CHECK(aclrtCreateStream(&stream));
        group_lock::lock_update_kernel<int32_t>
            <<<1, 0, stream>>>(d_update_count_, d_read_count_);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        NPU_CHECK(aclrtDestroyStream(stream));
        break;
      }

      // Failed to get lock, decrement and retry
      h_update_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  /**
   * @brief Release shared update lock
   */
  void unlock_update(aclrtStream stream) {
    group_lock::unlock_update_kernel<int32_t>
        <<<1, 0, stream>>>(d_update_count_);
    h_update_count_.fetch_sub(1, std::memory_order_release);
  }

  /**
   * @brief Acquire exclusive update_read lock
   */
  void lock_update_read() {
    // Lock unique flag (spin until we get it)
    bool expected = false;
    while (!h_unique_flag_.compare_exchange_weak(expected, true,
                                                 std::memory_order_acq_rel)) {
      expected = false;
    }

    // Ban update
    for (;;) {
      while (h_update_count_.load(std::memory_order_acquire)) {
      }
      h_read_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_update_count_.load(std::memory_order_acquire) == 0) {
        break;
      }
      h_read_count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    // Ban read (ensure only 1 reader - ourselves)
    for (;;) {
      while (h_read_count_.load(std::memory_order_acquire) > 1) {
      }
      h_update_count_.fetch_add(1, std::memory_order_acq_rel);
      if (h_read_count_.load(std::memory_order_acquire) == 1) {
        break;
      }
      h_update_count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    // Sync with device
    aclrtStream stream;
    NPU_CHECK(aclrtCreateStream(&stream));
    group_lock::lock_update_read_kernel<int32_t>
        <<<1, 0, stream>>>(d_update_count_, d_read_count_, d_unique_flag_);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    NPU_CHECK(aclrtDestroyStream(stream));
  }

  /**
   * @brief Release exclusive update_read lock
   */
  void unlock_update_read(aclrtStream stream) {
    group_lock::unlock_update_read_kernel<int32_t>
        <<<1, 0, stream>>>(d_update_count_, d_read_count_, d_unique_flag_);
    h_read_count_.fetch_sub(1, std::memory_order_release);
    h_update_count_.fetch_sub(1, std::memory_order_release);
    h_unique_flag_.store(false, std::memory_order_release);
  }

  /**
   * @brief Get current update count
   */
  int update_count() noexcept {
    int count = 0;
    int32_t* d_count;
    NPU_CHECK(aclrtMalloc((void**)&d_count, sizeof(int32_t),
                          ACL_MEM_MALLOC_NORMAL_ONLY));

    aclrtStream stream;
    NPU_CHECK(aclrtCreateStream(&stream));
    group_lock::update_count_kernel<int32_t>
        <<<1, 0, stream>>>(d_count, d_update_count_);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    NPU_CHECK(aclrtMemcpy(&count, sizeof(int32_t), d_count, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtFree(d_count));
    NPU_CHECK(aclrtDestroyStream(stream));

    return count;
  }

  /**
   * @brief Get current read count
   */
  int read_count() noexcept {
    int count = 0;
    int32_t* d_count;
    NPU_CHECK(aclrtMalloc((void**)&d_count, sizeof(int32_t),
                          ACL_MEM_MALLOC_NORMAL_ONLY));

    aclrtStream stream;
    NPU_CHECK(aclrtCreateStream(&stream));
    group_lock::read_count_kernel<int32_t>
        <<<1, 0, stream>>>(d_count, d_read_count_);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    NPU_CHECK(aclrtMemcpy(&count, sizeof(int32_t), d_count, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtFree(d_count));
    NPU_CHECK(aclrtDestroyStream(stream));

    return count;
  }

 private:
  // Host-side atomic counters
  std::atomic<int> h_update_count_;
  std::atomic<int> h_read_count_;
  std::atomic<bool> h_unique_flag_;

  // Device-side counters (for kernel sync if needed)
  int32_t* d_update_count_;
  int32_t* d_read_count_;
  int32_t* d_unique_flag_;
};

/**
 * @brief RAII shared read lock
 */
class read_shared_lock {
 public:
  read_shared_lock(const read_shared_lock&) = delete;
  read_shared_lock(read_shared_lock&&) = delete;

  read_shared_lock& operator=(const read_shared_lock&) = delete;
  read_shared_lock& operator=(read_shared_lock&&) = delete;

  explicit read_shared_lock(group_shared_mutex& mutex, aclrtStream stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_read();
    owns_ = true;
    stream_ = stream;
  }

  explicit read_shared_lock(group_shared_mutex& mutex, std::defer_lock_t,
                            aclrtStream stream = 0)
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~read_shared_lock() {
    if (owns_) {
      mutex_->unlock_read(stream_);
    }
  }

  void lock() noexcept {
    if (!owns_) {
      mutex_->lock_read();
      owns_ = true;
    }
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  aclrtStream stream_;
};

/**
 * @brief RAII shared update lock
 */
class update_shared_lock {
 public:
  update_shared_lock(const update_shared_lock&) = delete;
  update_shared_lock(update_shared_lock&&) = delete;

  update_shared_lock& operator=(const update_shared_lock&) = delete;
  update_shared_lock& operator=(update_shared_lock&&) = delete;

  explicit update_shared_lock(group_shared_mutex& mutex, aclrtStream stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_update();
    owns_ = true;
    stream_ = stream;
  }

  explicit update_shared_lock(group_shared_mutex& mutex, std::defer_lock_t,
                              aclrtStream stream = 0)
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~update_shared_lock() {
    if (owns_) {
      mutex_->unlock_update(stream_);
    }
  }

  void lock() noexcept {
    if (!owns_) {
      mutex_->lock_update();
      owns_ = true;
    }
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  aclrtStream stream_;
};

/**
 * @brief RAII exclusive update_read lock
 */
class update_read_lock {
 public:
  update_read_lock(const update_read_lock&) = delete;
  update_read_lock(update_read_lock&&) = delete;

  update_read_lock& operator=(const update_read_lock&) = delete;
  update_read_lock& operator=(update_read_lock&&) = delete;

  explicit update_read_lock(group_shared_mutex& mutex, aclrtStream stream = 0)
      : mutex_(&mutex) {
    mutex_->lock_update_read();
    owns_ = true;
    stream_ = stream;
  }

  explicit update_read_lock(group_shared_mutex& mutex, std::defer_lock_t,
                            aclrtStream stream = 0) noexcept
      : mutex_(&mutex), stream_(stream), owns_(false) {}

  ~update_read_lock() {
    if (owns_) {
      mutex_->unlock_update_read(stream_);
    }
  }

  void lock() {
    assert(!owns_ && "[update_read_lock] trying to lock twice!");
    mutex_->lock_update_read();
    owns_ = true;
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
  aclrtStream stream_;
};

// Alias for insert unique lock
using insert_unique_lock = update_read_lock;

}  // namespace hkv
}  // namespace npu
