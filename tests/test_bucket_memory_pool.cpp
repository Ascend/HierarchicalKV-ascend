/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

class BucketMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    init_env();
    size_t total_mem = 0;
    size_t free_mem = 0;
    constexpr size_t hbm_for_values = 1UL << 30;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
              ACL_ERROR_NONE);
    ASSERT_GT(free_mem, hbm_for_values)
        << "free HBM is not enough free:" << free_mem
        << " need:" << hbm_for_values;
    hbm_for_values_ = hbm_for_values;
  }

  void TearDown() override {
    // 清理环境变量
    unsetenv("HKV_NPU_ALLOC_CONF");
  }

  size_t hbm_for_values_ = 0;
};

// 测试默认配置（buckets_mem_pool=enable;page_table=2m）, 同时测试超大capacity场景
TEST_F(BucketMemoryPoolTest, test_default_config) {
  // 不设置环境变量，应该使用默认配置
  unsetenv("HKV_NPU_ALLOC_CONF");
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 512UL * 1024 * 1024;
  constexpr size_t max_capacity = init_capacity * 4;
  constexpr size_t key_num = init_capacity / 128;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_ * 16,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  // 插入数据验证功能正常
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试环境变量buckets_mem_pool=disable
TEST_F(BucketMemoryPoolTest, test_disable_memory_pool) {
  setenv("HKV_NPU_ALLOC_CONF", "buckets_mem_pool=disable", 1);
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 4;
  constexpr size_t key_num = 1024;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  // 验证功能正常（使用原有分配方式）
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试环境变量buckets_mem_pool=enable;page_table=1g
TEST_F(BucketMemoryPoolTest, test_enable_memory_pool_1g) {
  setenv("HKV_NPU_ALLOC_CONF", "buckets_mem_pool=enable;page_table=1g", 1);
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 4;
  constexpr size_t key_num = 1024;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  // 验证功能正常
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试环境变量buckets_mem_pool=enable;page_table=2m
TEST_F(BucketMemoryPoolTest, test_enable_memory_pool_2m) {
  setenv("HKV_NPU_ALLOC_CONF", "buckets_mem_pool=enable;page_table=2m", 1);
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 4;
  constexpr size_t key_num = 1024;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  // 验证功能正常
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试大小写不敏感
TEST_F(BucketMemoryPoolTest, test_case_insensitive) {
  setenv("HKV_NPU_ALLOC_CONF", "BUCKETS_MEM_POOL=ENABLE;PAGE_TABLE=2M", 1);
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 4;
  constexpr size_t key_num = 1024;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  // 验证功能正常
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试reserve接口在内存池方案下的扩容
TEST_F(BucketMemoryPoolTest, test_reserve_with_memory_pool) {
  setenv("HKV_NPU_ALLOC_CONF", "buckets_mem_pool=enable;page_table=2m", 1);
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 16;
  constexpr size_t key_num = 1024;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  
  // 插入数据
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  
  // 测试reserve扩容
  auto cur_capacity = table.capacity();
  EXPECT_EQ(cur_capacity, init_capacity);
  
  // 扩容到2倍
  table.reserve(cur_capacity + 1, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  cur_capacity = table.capacity();
  EXPECT_EQ(cur_capacity, init_capacity * 2);
  EXPECT_EQ(table.size(stream), key_num);
  
  // 验证数据完整性
  table.find(key_num, device_keys, device_values_ptr, device_found,
             nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
    }
  }
  EXPECT_EQ(found_num, key_num);
  // 继续扩容
  table.reserve(max_capacity, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  cur_capacity = table.capacity();
  EXPECT_EQ(cur_capacity, max_capacity);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试disable时忽略page_table配置
TEST_F(BucketMemoryPoolTest, test_disable_ignores_page_table) {
  setenv("HKV_NPU_ALLOC_CONF", "buckets_mem_pool=disable;page_table=1g", 1);
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 4;
  constexpr size_t key_num = 1024;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  // 验证功能正常（应该使用原有分配方式，忽略page_table配置）
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试空格处理
TEST_F(BucketMemoryPoolTest, test_whitespace_handling) {
  setenv("HKV_NPU_ALLOC_CONF", " buckets_mem_pool = enable ; page_table = 2m ", 1);
  
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 4;
  constexpr size_t key_num = 1024;
  
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = hbm_for_values_,
      .dim = dim,
      .io_by_cpu = false,
  };
  
  using Table = HashTable<K, V>;
  Table table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);
  EXPECT_EQ(table.capacity(), init_capacity);
  
  // 验证功能正常
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim, 1.0f);
  K* device_keys = nullptr;
  V* device_values = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * sizeof(V) * dim, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}
