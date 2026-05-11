/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <typeinfo>
#include "allocator.h"
#include "memory_pool.h"
#include "test_util.h"

using namespace community_test_util;
using namespace npu::hkv;

namespace {

/**
 * Wrapper around another allocator that prints debug messages.
 */
template <class Allocator>
struct DebugAllocator final
    : AllocatorBase<typename Allocator::type, DebugAllocator<Allocator>> {
  using type = typename Allocator::type;

  static constexpr const char* name{"DebugAllocator"};

  inline static type* alloc(size_t n, BaseAllocator* allocator,
                            aclrtStream stream = 0) {
    type* ptr{Allocator::alloc(n, allocator, stream)};
    std::cout << Allocator::name << "[type_name = " << typeid(type).name()
              << "]: " << static_cast<void*>(ptr) << " allocated = " << n
              << " x " << sizeof(type) << " bytes, stream = " << stream
              << '\n';
    return ptr;
  }

  inline static void free(type* ptr, BaseAllocator* allocator,
                          aclrtStream stream = 0) {
    Allocator::free(ptr, allocator, stream);
    std::cout << Allocator::name << "[type_name = " << typeid(type).name()
              << "]: " << static_cast<void*>(ptr)
              << " freed, stream = " << stream << '\n';
  }
};

void print_divider() {
  for (size_t i{0}; i < 80; ++i) {
    std::cout << '-';
  }
  std::cout << '\n';
}

void print_pool_options(const MemoryPoolOptions& opt) {
  print_divider();
  std::cout << "Memory Pool Configuration\n";
  print_divider();
  std::cout << "opt.max_stock   : " << opt.max_stock << " buffers\n";
  std::cout << "opt.max_pending : " << opt.max_pending << " buffers\n";
  print_divider();
  std::cout.flush();
}

constexpr MemoryPoolOptions kPoolOptions{
    3,  // max_stock
    5,  // max_pending
};

struct SomeType {
  int a;
  float b;
};

void expect_host_value(const SomeType& value, int a, float b) {
  EXPECT_EQ(value.a, a);
  EXPECT_FLOAT_EQ(value.b, b);
}

SomeType copy_from_device(const SomeType* device_ptr) {
  SomeType host_value{};
  EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(SomeType), device_ptr,
                        sizeof(SomeType), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  return host_value;
}

void copy_to_device(SomeType* device_ptr, const SomeType& host_value) {
  ASSERT_EQ(aclrtMemcpy(device_ptr, sizeof(SomeType), &host_value,
                        sizeof(SomeType), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
}

class MemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    init_env();
    allocator_ = std::make_unique<DefaultAllocator>();
  }

  std::unique_ptr<DefaultAllocator> allocator_;
};

void test_standard_allocator(DefaultAllocator* allocator) {
  using Allocator = DebugAllocator<StandardAllocator<SomeType>>;

  {
    auto ptr{Allocator::make_unique(1, allocator)};
    ASSERT_NE(ptr.get(), nullptr);

    ptr->a = 47;
    ptr->b = 11.0f;
    expect_host_value(*ptr, 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_unique(1, allocator, nullptr)};
    ASSERT_NE(ptr.get(), nullptr);

    ptr->a = 47;
    ptr->b = 11.0f;
    expect_host_value(*ptr, 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_shared(1, allocator)};
    ASSERT_NE(ptr.get(), nullptr);

    ptr->a = 47;
    ptr->b = 11.0f;
    expect_host_value(*ptr, 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_host_allocator(DefaultAllocator* allocator) {
  using Allocator = DebugAllocator<HostAllocator<SomeType>>;

  {
    auto ptr{Allocator::make_unique(1, allocator)};
    ASSERT_NE(ptr.get(), nullptr);

    ptr->a = 47;
    ptr->b = 11.0f;
    expect_host_value(*ptr, 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_unique(1, allocator, nullptr)};
    ASSERT_NE(ptr.get(), nullptr);

    ptr->a = 47;
    ptr->b = 11.0f;
    expect_host_value(*ptr, 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_shared(1, allocator)};
    ASSERT_NE(ptr.get(), nullptr);

    ptr->a = 47;
    ptr->b = 11.0f;
    expect_host_value(*ptr, 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_device_allocator(DefaultAllocator* allocator) {
  using Allocator = DebugAllocator<DeviceAllocator<SomeType>>;

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  {
    auto ptr{Allocator::make_unique(1, allocator)};
    ASSERT_NE(ptr.get(), nullptr);

    copy_to_device(ptr.get(), SomeType{47, 11.0f});
    expect_host_value(copy_from_device(ptr.get()), 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_unique(1, allocator, stream)};
    ASSERT_NE(ptr.get(), nullptr);

    copy_to_device(ptr.get(), SomeType{47, 11.0f});
    expect_host_value(copy_from_device(ptr.get()), 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr{Allocator::make_shared(1, allocator, stream)};
    ASSERT_NE(ptr.get(), nullptr);

    copy_to_device(ptr.get(), SomeType{47, 11.0f});
    expect_host_value(copy_from_device(ptr.get()), 47, 11.0f);

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

void test_borrow_return_with_context(DefaultAllocator* allocator,
                                     bool use_custom_stream) {
  aclrtStream stream{0};
  if (use_custom_stream) {
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  }

  print_pool_options(kPoolOptions);

  {
    MemoryPool<DebugAllocator<DeviceAllocator<SomeType>>> pool(kPoolOptions,
                                                               allocator);
    const size_t buffer_size{256L * 1024};

    // Initial status.
    std::cout << ".:: Initial state ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (unique ptr).
    {
      auto buffer{pool.get_unique(buffer_size, stream)};
      std::cout << ".:: Borrow 1 (unique) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (unique) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (shared ptr).
    {
      auto buffer{pool.get_shared(buffer_size, stream)};
      std::cout << ".:: Borrow 1 (shared) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 1 (shared) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow static workspace with less than `max_stock` buffers.
    {
      auto ws{pool.get_workspace<2>(buffer_size, stream)};
      std::cout << ".:: Borrow 2 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 2 (static) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 2);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow workspace that exceeds base pool size. Possible results:
    // 1. If this thread is slower than the driver.
    //    Upon return we will see a partial deallocation before inserting the
    //    last buffer into the pending queue.
    // 2. If the driver is slower than this thread queuing/querying events.
    //    Either 0-3 buffers in stock partial deallocation,
    //    1-5 buffers pending. Hence there is no good way to check.
    {
      auto ws{pool.get_workspace<6>(buffer_size, stream)};
      std::cout << ".:: Borrow 6 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Return 6 (static) ::.\n" << pool << std::endl;
    ASSERT_GE(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 3);
    ASSERT_EQ(pool.num_pending(), 0);

    // Pin 1 and deplete stock.
    {
      auto ws{pool.get_workspace<1>(buffer_size, stream)};
      pool.deplete_stock();
      std::cout << ".:: Deplete stock ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);
    }
    std::cout << ".:: Deplete stock ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 1);

    // Await unfinished GPU work (ensure stable situation).
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Increase stock to 3 buffers.
    {
      auto ws{pool.get_workspace<3>(buffer_size, stream)};
    }
    pool.await_pending(stream);
    ASSERT_EQ(pool.current_stock(), 3);
    ASSERT_EQ(pool.num_pending(), 0);

    // Pin 1 of the 3 buffers and release it to make it pending.
    {
      auto ws{pool.get_workspace<1>(buffer_size, stream)};
    }
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 1);
    std::cout << ".:: Ensure 2 stock + 1 pending situation ::.\n"
              << pool << std::endl;

    // Borrow a buffer that is smaller than the current buffer size.
    {
      auto ws{pool.get_unique(buffer_size / 2, stream)};
      std::cout << ".:: Borrow 1 (smaller) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 1);
      ASSERT_EQ(pool.num_pending(), 1);
    }
    std::cout << ".:: Return 1 (smaller) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 2);

    // Borrow a buffer that is bigger than the current buffer size. This will
    // evict the stock buffers which are smaller, but will not concern the
    // buffers that are still pending.
    {
      auto ws{pool.get_unique(buffer_size + 37, stream)};
      std::cout << ".:: Borrow 1 (bigger) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 2);
    }
    std::cout << ".:: Return 1 (bigger) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 3);

    // Because there are now pending buffers that are too small, they will be
    // cleared once the associated work has been completed.
    pool.await_pending(stream);
    std::cout << ".:: Await pending ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  if (stream != 0) {
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }
}

void test_borrow_return_lost_context(DefaultAllocator* allocator) {
  print_pool_options(kPoolOptions);

  {
    MemoryPool<DebugAllocator<DeviceAllocator<SomeType>>> pool(kPoolOptions,
                                                               allocator);
    const size_t buffer_size{256L * 1024};

    // Initial status.
    std::cout << ".:: Initial state ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (unique ptr).
    {
      aclrtStream stream = nullptr;
      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

      auto buffer{pool.get_unique(buffer_size, stream)};
      std::cout << ".:: Borrow 1 (unique) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);

      ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
    std::cout << ".:: Return 1 (unique) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow and return one buffer (shared ptr).
    {
      aclrtStream stream = nullptr;
      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

      auto buffer{pool.get_shared(buffer_size)};
      std::cout << ".:: Borrow 1 (shared) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);

      ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
    std::cout << ".:: Return 1 (shared) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow static workspace with less than `max_stock` buffers.
    {
      aclrtStream stream = nullptr;
      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

      auto ws{pool.get_workspace<2>(buffer_size)};
      std::cout << ".:: Borrow 2 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);

      ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
    std::cout << ".:: Return 2 (static) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow dynamic workspace with less than `max_stock` buffers.
    {
      aclrtStream stream = nullptr;
      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

      auto ws{pool.get_workspace(2, buffer_size)};
      std::cout << ".:: Borrow 2 (dynamic) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);

      ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }

    std::cout << ".:: Return 2 (dynamic) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Await unfinished GPU work (shouldn't change anything).
    pool.await_pending();
    std::cout << ".:: Await pending (shouldn't change anything) ::.\n"
              << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 2);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow workspace that exceeds base pool size.
    {
      aclrtStream stream = nullptr;
      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

      auto ws{pool.get_workspace<6>(buffer_size)};
      std::cout << ".:: Borrow 6 (static) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);

      ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
    std::cout << ".:: Return 6 (static) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), kPoolOptions.max_stock);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow a buffer that is smaller than the current buffer size.
    {
      aclrtStream stream = nullptr;
      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

      auto ws{pool.get_unique(buffer_size / 2)};
      std::cout << ".:: Borrow 1 (smaller) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), kPoolOptions.max_stock - 1);
      ASSERT_EQ(pool.num_pending(), 0);

      ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
    std::cout << ".:: Return 1 (smaller) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), kPoolOptions.max_stock);
    ASSERT_EQ(pool.num_pending(), 0);

    // Borrow a buffer that is bigger than the current buffer size.
    {
      aclrtStream stream = nullptr;
      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

      auto ws{pool.get_unique(buffer_size + 37)};
      std::cout << ".:: Borrow 1 (bigger) ::.\n" << pool << std::endl;
      ASSERT_EQ(pool.current_stock(), 0);
      ASSERT_EQ(pool.num_pending(), 0);

      ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
    std::cout << ".:: Return 1 (bigger) ::.\n" << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 1);
    ASSERT_EQ(pool.num_pending(), 0);
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

TEST_F(MemoryPoolTest, test_borrow_return_lost_context) {
  std::cout << "Unfortunately, there is currently no reliable way to test "
               "safely whether a\n"
            << "stream is alive. Keeping the test around for manual tests.\n";
  if (false) {
    test_borrow_return_lost_context(allocator_.get());
  }
}

}  // namespace
