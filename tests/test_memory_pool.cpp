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

#include <memory>
#include <gtest/gtest.h>
#include "memory_pool.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace test_util;

namespace {

constexpr size_t kBufferSize = 256UL * 1024;
constexpr MemoryPoolOptions kPoolOptions{
    3,   // max_stock
    5,   // max_pending
};

struct SomeType {
  int a;
  float b;
};

class MemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    init_env();
    allocator_ = std::make_unique<DefaultAllocator>();
  }

  std::unique_ptr<DefaultAllocator> allocator_;
};

void expect_host_value(const SomeType& value, int a, float b) {
  EXPECT_EQ(value.a, a);
  EXPECT_FLOAT_EQ(value.b, b);
}

SomeType copy_from_device(const SomeType* device_ptr) {
  SomeType host_value{};
  auto ret = aclrtMemcpy(&host_value, sizeof(SomeType), device_ptr,
                         sizeof(SomeType), ACL_MEMCPY_DEVICE_TO_HOST);
  EXPECT_EQ(ret, ACL_ERROR_NONE);
  return host_value;
}

void copy_to_device(SomeType* device_ptr, const SomeType& host_value) {
  ASSERT_EQ(aclrtMemcpy(device_ptr, sizeof(SomeType), &host_value,
                        sizeof(SomeType), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
}

void test_standard_allocator(DefaultAllocator* allocator) {
  using Allocator = StandardAllocator<SomeType>;

  {
    auto ptr = Allocator::make_unique(1, allocator);
    ASSERT_NE(ptr.get(), nullptr);
    ptr->a = 47;
    ptr->b = 11.0f;
    expect_host_value(*ptr, 47, 11.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_unique(1, allocator, nullptr);
    ASSERT_NE(ptr.get(), nullptr);
    ptr->a = 48;
    ptr->b = 12.0f;
    expect_host_value(*ptr, 48, 12.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_shared(1, allocator);
    ASSERT_NE(ptr.get(), nullptr);
    ptr->a = 49;
    ptr->b = 13.0f;
    expect_host_value(*ptr, 49, 13.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_host_allocator(DefaultAllocator* allocator) {
  using Allocator = HostAllocator<SomeType>;

  {
    auto ptr = Allocator::make_unique(1, allocator);
    ASSERT_NE(ptr.get(), nullptr);
    ptr->a = 57;
    ptr->b = 21.0f;
    expect_host_value(*ptr, 57, 21.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_unique(1, allocator, nullptr);
    ASSERT_NE(ptr.get(), nullptr);
    ptr->a = 58;
    ptr->b = 22.0f;
    expect_host_value(*ptr, 58, 22.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_shared(1, allocator);
    ASSERT_NE(ptr.get(), nullptr);
    ptr->a = 59;
    ptr->b = 23.0f;
    expect_host_value(*ptr, 59, 23.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_device_allocator(DefaultAllocator* allocator) {
  using Allocator = DeviceAllocator<SomeType>;

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  {
    auto ptr = Allocator::make_unique(1, allocator);
    ASSERT_NE(ptr.get(), nullptr);
    copy_to_device(ptr.get(), SomeType{67, 31.0f});
    auto value = copy_from_device(ptr.get());
    expect_host_value(value, 67, 31.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_unique(1, allocator, stream);
    ASSERT_NE(ptr.get(), nullptr);
    copy_to_device(ptr.get(), SomeType{68, 32.0f});
    auto value = copy_from_device(ptr.get());
    expect_host_value(value, 68, 32.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_shared(1, allocator, stream);
    ASSERT_NE(ptr.get(), nullptr);
    copy_to_device(ptr.get(), SomeType{69, 33.0f});
    auto value = copy_from_device(ptr.get());
    expect_host_value(value, 69, 33.0f);
    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

void test_borrow_return_with_context(DefaultAllocator* allocator,
                                     bool use_custom_stream) {
  aclrtStream stream = nullptr;
  if (use_custom_stream) {
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  }

  MemoryPool<DeviceAllocator<char>> pool(kPoolOptions, allocator);

  EXPECT_EQ(pool.current_stock(), 0);
  EXPECT_EQ(pool.num_pending(), 0);

  {
    auto buffer = pool.get_unique(kBufferSize, stream);
    ASSERT_NE(buffer.get(), nullptr);
    EXPECT_EQ(pool.current_stock(), 0);
    EXPECT_EQ(pool.num_pending(), 0);
  }

  EXPECT_EQ(pool.current_stock(), 0);
  EXPECT_EQ(pool.num_pending(), 1);

  pool.await_pending(stream);
  EXPECT_EQ(pool.current_stock(), 1);
  EXPECT_EQ(pool.num_pending(), 0);

  {
    auto buffer = pool.get_shared(kBufferSize, stream);
    ASSERT_NE(buffer.get(), nullptr);
    EXPECT_EQ(pool.current_stock(), 0);
    EXPECT_EQ(pool.num_pending(), 0);
  }

  EXPECT_EQ(pool.current_stock(), 0);
  EXPECT_EQ(pool.num_pending(), 1);

  pool.await_pending(stream);
  EXPECT_EQ(pool.current_stock(), 1);
  EXPECT_EQ(pool.num_pending(), 0);

  {
    auto ws = pool.get_workspace<2>(kBufferSize, stream);
    EXPECT_EQ(pool.current_stock(), 0);
    EXPECT_EQ(pool.num_pending(), 0);
  }

  EXPECT_EQ(pool.current_stock(), 0);
  EXPECT_EQ(pool.num_pending(), 2);

  pool.await_pending(stream);
  EXPECT_EQ(pool.current_stock(), 2);
  EXPECT_EQ(pool.num_pending(), 0);

  {
    auto ws = pool.get_workspace<6>(kBufferSize, stream);
    EXPECT_EQ(pool.current_stock(), 0);
    EXPECT_EQ(pool.num_pending(), 0);
  }

  EXPECT_GE(pool.num_pending(), 1);

  pool.await_pending(stream);
  EXPECT_EQ(pool.current_stock(), 3);
  EXPECT_EQ(pool.num_pending(), 0);

  {
    auto ws = pool.get_workspace<1>(kBufferSize, stream);
    pool.deplete_stock();
    EXPECT_EQ(pool.current_stock(), 0);
    EXPECT_EQ(pool.num_pending(), 0);
  }

  EXPECT_EQ(pool.current_stock(), 0);
  EXPECT_EQ(pool.num_pending(), 1);

  pool.await_pending(stream);
  EXPECT_EQ(pool.current_stock(), 1);
  EXPECT_EQ(pool.num_pending(), 0);

  {
    auto ws = pool.get_workspace<3>(kBufferSize, stream);
  }
  pool.await_pending(stream);
  EXPECT_EQ(pool.current_stock(), 3);
  EXPECT_EQ(pool.num_pending(), 0);

  {
    auto ws = pool.get_workspace<1>(kBufferSize, stream);
  }
  EXPECT_EQ(pool.current_stock(), 2);
  EXPECT_EQ(pool.num_pending(), 1);

  {
    auto ws = pool.get_unique(kBufferSize / 2, stream);
    ASSERT_NE(ws.get(), nullptr);
    EXPECT_EQ(pool.current_stock(), 1);
    EXPECT_EQ(pool.num_pending(), 1);
  }
  EXPECT_EQ(pool.current_stock(), 1);
  EXPECT_EQ(pool.num_pending(), 2);

  {
    auto ws = pool.get_unique(kBufferSize + 37, stream);
    ASSERT_NE(ws.get(), nullptr);
    EXPECT_EQ(pool.current_stock(), 0);
    EXPECT_EQ(pool.num_pending(), 2);
  }
  EXPECT_EQ(pool.current_stock(), 0);
  EXPECT_EQ(pool.num_pending(), 3);

  pool.await_pending(stream);
  EXPECT_EQ(pool.current_stock(), 1);
  EXPECT_EQ(pool.num_pending(), 0);

  if (stream != nullptr) {
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }
}

TEST_F(MemoryPoolTest, standard_allocator) {
  test_standard_allocator(allocator_.get());
}

TEST_F(MemoryPoolTest, host_allocator) {
  test_host_allocator(allocator_.get());
}

TEST_F(MemoryPoolTest, device_allocator) {
  test_device_allocator(allocator_.get());
}

TEST_F(MemoryPoolTest, borrow_return_default_context) {
  test_borrow_return_with_context(allocator_.get(), false);
}

TEST_F(MemoryPoolTest, borrow_return_custom_context) {
  test_borrow_return_with_context(allocator_.get(), true);
}

}  // namespace
