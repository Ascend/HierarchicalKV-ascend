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

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <system_error>
#include <thread>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "group_lock.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace test_util;
using namespace std::chrono_literals;

class GroupSharedMutexTest : public ::testing::Test {
 protected:
  void SetUp() override { init_env(); }
};

// Test the basic functionality of the group_shared_mutex
TEST_F(GroupSharedMutexTest, BasicFunctionality) {
  group_shared_mutex mutex;
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);

  {
    // Multiple reads can acquire the lock simultaneously
    read_shared_lock read1(mutex);
    ASSERT_EQ(mutex.read_count(), 1);
    read_shared_lock read2(mutex);
    ASSERT_EQ(mutex.read_count(), 2);
  }
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);

  {
    // A update is blocked by the reads
    update_shared_lock update(mutex, std::defer_lock);
    EXPECT_FALSE(update.owns_lock());
    ASSERT_EQ(mutex.read_count(), 0);
    ASSERT_EQ(mutex.update_count(), 0);
    update.lock();
    ASSERT_EQ(mutex.read_count(), 0);
    ASSERT_EQ(mutex.update_count(), 1);
    EXPECT_TRUE(update.owns_lock());
  }
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);

  // A unique lock is also blocked by the reads
  {
    update_read_lock unique(mutex, std::defer_lock);
    ASSERT_EQ(mutex.read_count(), 0);
    ASSERT_EQ(mutex.update_count(), 0);
    EXPECT_FALSE(unique.owns_lock());
    unique.lock();
    EXPECT_TRUE(unique.owns_lock());
    ASSERT_EQ(mutex.read_count(), 1);
    ASSERT_EQ(mutex.update_count(), 1);
  }
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);

  // Multiple reads can acquire the lock simultaneously, with user created
  // stream
  aclrtStream stream;
  NPU_CHECK(aclrtCreateStream(&stream));
  {
    read_shared_lock read1(mutex, stream);
    ASSERT_EQ(mutex.read_count(), 1);
    read_shared_lock read2(mutex, stream);
    ASSERT_EQ(mutex.read_count(), 2);
  }
  NPU_CHECK(aclrtSynchronizeStream(stream));
  ASSERT_EQ(mutex.read_count(), 0);
  ASSERT_EQ(mutex.update_count(), 0);
  NPU_CHECK(aclrtDestroyStream(stream));
}

TEST_F(GroupSharedMutexTest, AdvancedFunctionalityMultiStream) {
  group_shared_mutex mutex;
  std::atomic<bool> multiple_read{false};
  std::atomic<bool> multiple_update{false};

  // Get device from test environment
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = 0;
  try {
    device_id = (device_id_env != nullptr) ? std::stoi(device_id_env) : 0;
  } catch (...) {
    device_id = 0;
    std::cout << "set env HKV_TEST_DEVICE error, using default device_id 0"
              << std::endl;
  }

  // Test multiple reads
  std::vector<std::thread> reads;
  for (int i = 0; i < 50; ++i) {
    reads.emplace_back([&]() {
      // Each thread needs its own stream with valid context
      aclrtStream stream;
      NPU_CHECK(aclrtSetDevice(device_id));
      NPU_CHECK(aclrtCreateStream(&stream));
      {
        read_shared_lock read(mutex, stream);
        EXPECT_TRUE(mutex.read_count() > 0);
        if (mutex.read_count() > 1) multiple_read.store(true);
        std::this_thread::sleep_for(1000ms);
        ASSERT_EQ(mutex.update_count(), 0);
      }
      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtDestroyStream(stream));
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  // Test multiple updates
  std::vector<std::thread> updates;
  for (int i = 0; i < 50; ++i) {
    updates.emplace_back([&]() {
      aclrtStream stream;
      NPU_CHECK(aclrtSetDevice(device_id));
      NPU_CHECK(aclrtCreateStream(&stream));
      {
        update_shared_lock update(mutex, stream);
        EXPECT_TRUE(mutex.update_count() > 0);
        if (mutex.update_count() > 1) multiple_update.store(true);
        std::this_thread::sleep_for(1000ms);
        ASSERT_EQ(mutex.read_count(), 0);
      }

      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtDestroyStream(stream));
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  // Test multiple uniques
  std::vector<std::thread> uniques;
  for (int i = 0; i < 50; ++i) {
    uniques.emplace_back([&]() {
      aclrtStream stream;
      NPU_CHECK(aclrtSetDevice(device_id));
      NPU_CHECK(aclrtCreateStream(&stream));
      {
        update_read_lock unique(mutex, stream);
        ASSERT_EQ(mutex.read_count(), 1);
        ASSERT_EQ(mutex.update_count(), 1);
        std::this_thread::sleep_for(100ms);
      }
      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtDestroyStream(stream));
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  for (auto& th : reads) {
    th.join();
  }

  for (auto& th : updates) {
    th.join();
  }

  for (auto& th : uniques) {
    th.join();
  }

  EXPECT_TRUE(multiple_update.load());
  EXPECT_TRUE(multiple_read.load());
}

TEST_F(GroupSharedMutexTest, AdvancedFunctionalitySingleStream) {
  group_shared_mutex mutex;
  std::atomic<bool> multiple_read{false};
  std::atomic<bool> multiple_update{false};

  // Get device from test environment
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = 0;
  try {
    device_id = (device_id_env != nullptr) ? std::stoi(device_id_env) : 0;
  } catch (...) {
    device_id = 0;
    std::cout << "set env HKV_TEST_DEVICE error, using default device_id 0"
              << std::endl;
  }

  // Test multiple reads
  std::vector<std::thread> reads;
  for (int i = 0; i < 50; ++i) {
    reads.emplace_back([&]() {
      NPU_CHECK(aclrtSetDevice(device_id));
      {
        read_shared_lock read(mutex);
        EXPECT_TRUE(mutex.read_count() > 0);
        if (mutex.read_count() > 1) multiple_read.store(true);
        std::this_thread::sleep_for(1000ms);
        ASSERT_EQ(mutex.update_count(), 0);
      }
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  // Test multiple updates
  std::vector<std::thread> updates;
  for (int i = 0; i < 50; ++i) {
    updates.emplace_back([&]() {
      NPU_CHECK(aclrtSetDevice(device_id));
      {
        update_shared_lock update(mutex);
        EXPECT_TRUE(mutex.update_count() > 0);
        if (mutex.update_count() > 1) multiple_update.store(true);
        std::this_thread::sleep_for(1000ms);
        ASSERT_EQ(mutex.read_count(), 0);
      }
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  // Test multiple uniques
  std::vector<std::thread> uniques;
  for (int i = 0; i < 50; ++i) {
    uniques.emplace_back([&]() {
      NPU_CHECK(aclrtSetDevice(device_id));
      {
        update_read_lock unique(mutex);
        ASSERT_EQ(mutex.read_count(), 1);
        ASSERT_EQ(mutex.update_count(), 1);
        std::this_thread::sleep_for(100ms);
      }
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  for (auto& th : reads) {
    th.join();
  }

  for (auto& th : updates) {
    th.join();
  }

  for (auto& th : uniques) {
    th.join();
  }

  EXPECT_TRUE(multiple_update.load());
  EXPECT_TRUE(multiple_read.load());
}

TEST_F(GroupSharedMutexTest, LockExclusivity) {
  group_shared_mutex mutex;
  std::atomic<bool> read_held{false};
  std::atomic<bool> update_held{false};

  // Test: When update lock is held, read lock should be blocked
  {
    aclrtStream stream;
    NPU_CHECK(aclrtCreateStream(&stream));
    {
      update_shared_lock update(mutex, stream);
      EXPECT_EQ(mutex.read_count(), 0);
      EXPECT_EQ(mutex.update_count(), 1);
    }
    NPU_CHECK(aclrtSynchronizeStream(stream));
    NPU_CHECK(aclrtDestroyStream(stream));
  }

  // Test: When read lock is held, update lock should be blocked
  {
    aclrtStream stream;
    NPU_CHECK(aclrtCreateStream(&stream));
    {
      read_shared_lock read(mutex, stream);
      EXPECT_EQ(mutex.update_count(), 0);
      EXPECT_EQ(mutex.read_count(), 1);
    }

    NPU_CHECK(aclrtSynchronizeStream(stream));
    NPU_CHECK(aclrtDestroyStream(stream));
  }

  // Test: When update_read lock is held, both should be blocked
  {
    aclrtStream stream;
    NPU_CHECK(aclrtCreateStream(&stream));
    {
      update_read_lock unique(mutex, stream);
      EXPECT_EQ(mutex.read_count(), 1);
      EXPECT_EQ(mutex.update_count(), 1);
    }
    NPU_CHECK(aclrtSynchronizeStream(stream));
    NPU_CHECK(aclrtDestroyStream(stream));
  }
}

TEST_F(GroupSharedMutexTest, MultiThreadedConcurrent) {
  group_shared_mutex mutex;
  const int NUM_THREADS = 10;
  std::atomic<int> read_acquired{0};
  std::atomic<int> update_acquired{0};
  std::atomic<int> unique_acquired{0};

  // Get device from test environment
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = 0;
  try {
    device_id = (device_id_env != nullptr) ? std::stoi(device_id_env) : 0;
  } catch (...) {
    device_id = 0;
    std::cout << "set env HKV_TEST_DEVICE error, using default device_id 0"
              << std::endl;
  }

  std::vector<std::thread> threads;

  // Create multiple readers
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&]() {
      aclrtStream stream;
      NPU_CHECK(aclrtSetDevice(device_id));
      NPU_CHECK(aclrtCreateStream(&stream));
      {
        read_shared_lock read(mutex, stream);
        read_acquired.fetch_add(1);
        std::this_thread::sleep_for(100ms);
      }
      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtDestroyStream(stream));
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  // Create multiple updaters
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&]() {
      aclrtStream stream;
      NPU_CHECK(aclrtSetDevice(device_id));
      NPU_CHECK(aclrtCreateStream(&stream));
      {
        update_shared_lock update(mutex, stream);
        update_acquired.fetch_add(1);
        std::this_thread::sleep_for(100ms);
      }
      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtDestroyStream(stream));
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  // Create multiple unique locks
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&]() {
      aclrtStream stream;
      NPU_CHECK(aclrtSetDevice(device_id));
      NPU_CHECK(aclrtCreateStream(&stream));
      {
        update_read_lock unique(mutex, stream);
        unique_acquired.fetch_add(1);
        std::this_thread::sleep_for(100ms);
      }
      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtDestroyStream(stream));
      NPU_CHECK(aclrtResetDevice(device_id));
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(read_acquired.load(), NUM_THREADS);
  EXPECT_EQ(update_acquired.load(), NUM_THREADS);
  EXPECT_EQ(unique_acquired.load(), NUM_THREADS);
}
