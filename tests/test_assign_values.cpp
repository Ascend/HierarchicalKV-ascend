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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

// 测试夹具类，用于复用测试初始化和清理逻辑
class AssignValuesTest : public ::testing::Test {
 protected:
  static constexpr size_t hbm_for_values = 1UL << 30;
  static constexpr size_t init_capacity = 128UL * 1024;

  void SetUp() override {
    init_env();

    size_t total_mem = 0;
    size_t free_mem = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
              ACL_ERROR_NONE);
    ASSERT_GT(free_mem, hbm_for_values)
        << "free HBM is not enough free:" << free_mem
        << " need:" << hbm_for_values;

    ASSERT_EQ(aclrtCreateStream(&stream_), ACL_ERROR_NONE);
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
  }

  // 辅助函数：分配设备内存
  template <typename T>
  T* alloc_device_mem(size_t count) {
    T* ptr = nullptr;
    EXPECT_EQ(aclrtMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    return ptr;
  }

  // 辅助函数：释放设备内存
  template <typename T>
  void free_device_mem(T* ptr) {
    if (ptr != nullptr) {
      EXPECT_EQ(aclrtFree(ptr), ACL_ERROR_NONE);
    }
  }

  // 辅助函数：主机到设备拷贝
  template <typename T>
  void copy_to_device(T* dst, const T* src, size_t count) {
    EXPECT_EQ(aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  // 辅助函数：设备到主机拷贝
  template <typename T>
  void copy_to_host(T* dst, const T* src, size_t count) {
    EXPECT_EQ(aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
  }

  aclrtStream stream_ = nullptr;
};

// 测试1：基本功能测试 - 插入后更新 values

TEST_F(AssignValuesTest, basic_function) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1UL * 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  // 插入数据
  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 更新 values 为新值（原值 + 1.0）
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 1.0f;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_new_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 find 接口验证 values 已更新
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(key_num * dim);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> real_values(key_num * dim, 0);
  copy_to_host(real_values.data(), device_values_buffer, key_num * dim);
  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(real_values[i * dim + j], new_values[i * dim + j])
            << "Value at key index " << i << " dim " << j
            << " should be updated";
      }
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_values_buffer);
}

// 测试2：空表测试 - 对空表执行 assign_values 不会崩溃
TEST_F(AssignValuesTest, empty_table) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 100;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);

  // 在空表上执行 assign_values 应该不会崩溃
  table.assign_values(key_num, device_keys, device_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);

  free_device_mem(device_keys);
  free_device_mem(device_values);
}

// 测试3：边界情况 - n=0 时不崩溃
TEST_F(AssignValuesTest, zero_keys) {
  constexpr size_t dim = 8;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V> table;
  table.init(options);

  // n=0 时调用 assign_values 应该直接返回，不崩溃
  table.assign_values(0, nullptr, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);
}

// 测试4：单个 key 测试 - n=1
TEST_F(AssignValuesTest, single_key) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kEpochLfu> table;
  table.init(options);
  table.set_global_epoch(1);

  K host_key = 12345;
  S host_score = 100;
  vector<V> host_values(dim, 1.5f);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, &host_key, key_num);
  copy_to_device(device_values, host_values.data(), dim);
  copy_to_device(device_scores, &host_score, key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 更新 value
  vector<V> new_values(dim, 9.9f);
  copy_to_device(device_values, new_values.data(), dim);

  table.assign_values(key_num, device_keys, device_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证 value 已更新
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(dim);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  bool host_found = false;
  copy_to_host(&host_found, device_found, 1);
  EXPECT_TRUE(host_found);

  vector<V> real_values(dim, 0);
  copy_to_host(real_values.data(), device_values_buffer, dim);
  for (size_t j = 0; j < dim; j++) {
    EXPECT_FLOAT_EQ(real_values[j], new_values[j])
        << "Value at dim " << j << " should be updated";
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_values_buffer);
}

// 测试5：部分 keys 存在测试 - 只更新存在的 keys 的 values
TEST_F(AssignValuesTest, partial_keys_exist) {
  constexpr size_t dim = 8;
  constexpr size_t insert_key_num = 500;
  constexpr size_t update_key_num = 1000;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  // 插入前 500 个 keys (1-500)
  vector<K> insert_keys(insert_key_num, 0);
  vector<V> insert_values(insert_key_num * dim, 0);
  vector<S> insert_scores(insert_key_num, 0);
  create_continuous_keys<K, S, V, dim>(insert_keys.data(), insert_scores.data(),
                                       insert_values.data(), insert_key_num, 1);

  K* device_insert_keys = alloc_device_mem<K>(insert_key_num);
  V* device_insert_values = alloc_device_mem<V>(insert_key_num * dim);
  S* device_insert_scores = alloc_device_mem<S>(insert_key_num);

  copy_to_device(device_insert_keys, insert_keys.data(), insert_key_num);
  copy_to_device(device_insert_values, insert_values.data(),
                 insert_key_num * dim);
  copy_to_device(device_insert_scores, insert_scores.data(), insert_key_num);

  table.insert_or_assign(insert_key_num, device_insert_keys,
                         device_insert_values, device_insert_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), insert_key_num);

  // 尝试更新 1000 个 keys (1-1000)，只有前 500 个存在
  vector<K> update_keys(update_key_num, 0);
  vector<V> update_values(update_key_num * dim, 0);
  create_continuous_keys<K, S, V, dim>(update_keys.data(), nullptr,
                                       update_values.data(), update_key_num, 1);

  // 设置新 values（原值 + 100.0）
  for (size_t i = 0; i < update_key_num * dim; i++) {
    update_values[i] = 100.0f + static_cast<V>(i % dim);
  }

  K* device_update_keys = alloc_device_mem<K>(update_key_num);
  V* device_update_values = alloc_device_mem<V>(update_key_num * dim);

  copy_to_device(device_update_keys, update_keys.data(), update_key_num);
  copy_to_device(device_update_values, update_values.data(),
                 update_key_num * dim);

  // 执行 assign_values，不存在的 keys 应该被忽略
  table.assign_values(update_key_num, device_update_keys, device_update_values,
                      stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), insert_key_num);

  // 验证存在的 keys 的 values 已更新（仅验证前 500 个）
  V** device_values_ptr = alloc_device_mem<V*>(insert_key_num);
  bool* device_found = alloc_device_mem<bool>(insert_key_num);
  V* device_values_buffer = alloc_device_mem<V>(insert_key_num * dim);

  table.find(insert_key_num, device_insert_keys, device_values_ptr,
             device_found, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, insert_key_num,
                stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(insert_key_num);
  copy_to_host(host_found.get(), device_found, insert_key_num);
  vector<V> real_values(insert_key_num * dim, 0);
  copy_to_host(real_values.data(), device_values_buffer, insert_key_num * dim);

  size_t found_count = 0;
  for (size_t i = 0; i < insert_key_num; i++) {
    if (host_found[i]) {
      found_count++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(real_values[i * dim + j], update_values[i * dim + j])
            << "Value at index " << i << " dim " << j << " should be updated";
      }
    }
  }
  EXPECT_EQ(found_count, insert_key_num);

  free_device_mem(device_insert_keys);
  free_device_mem(device_insert_values);
  free_device_mem(device_insert_scores);
  free_device_mem(device_update_keys);
  free_device_mem(device_update_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_values_buffer);
}

// 测试6：随机 keys 测试
TEST_F(AssignValuesTest, random_keys) {
  constexpr size_t dim = 16;
  constexpr size_t key_num = 2048;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                              host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 更新 values
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] * 2.0f + 1.0f;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_new_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(key_num * dim);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);
  vector<V> real_values(key_num * dim);
  copy_to_host(real_values.data(), device_values_buffer, key_num * dim);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(real_values[i * dim + j], new_values[i * dim + j]);
      }
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_values_buffer);
}

// 测试7：大规模数据测试
TEST_F(AssignValuesTest, large_scale) {
  constexpr size_t dim = 32;
  constexpr size_t key_num = 64UL * 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 更新 values
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 10.0f;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_new_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
}

// 测试8：不同 dim 测试 - dim=4
TEST_F(AssignValuesTest, small_dim) {
  constexpr size_t dim = 4;
  constexpr size_t key_num = 512;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 更新 values
  vector<V> new_values(key_num * dim, 9.99f);
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_new_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) found_num++;
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
}

// 测试9：不同 dim 测试 - dim=128
TEST_F(AssignValuesTest, large_dim) {
  constexpr size_t dim = 128;
  constexpr size_t key_num = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 更新 values（原值 * 3）
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] * 3.0f;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_new_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(key_num * dim);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);
  vector<V> real_values(key_num * dim, 0);
  copy_to_host(real_values.data(), device_values_buffer, key_num * dim);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(real_values[i * dim + j], new_values[i * dim + j])
            << "Value at index " << i << " dim " << j << " should be updated";
      }
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_values_buffer);
}

// 测试10：多次更新 values 测试
TEST_F(AssignValuesTest, multiple_updates) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 256;
  constexpr size_t update_times = 5;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 多次更新 values
  for (size_t t = 0; t < update_times; t++) {
    vector<V> new_values(key_num * dim);
    for (size_t i = 0; i < key_num * dim; i++) {
      new_values[i] = static_cast<V>((t + 1) * 1000 + (i % dim));
    }
    copy_to_device(device_values, new_values.data(), key_num * dim);

    table.assign_values(key_num, device_keys, device_values, stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

    // 验证每次更新后 keys 仍然存在
    V** device_values_ptr = alloc_device_mem<V*>(key_num);
    bool* device_found = alloc_device_mem<bool>(key_num);

    table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
               stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

    auto host_found = std::make_unique<bool[]>(key_num);
    copy_to_host(host_found.get(), device_found, key_num);

    size_t found_num = 0;
    for (size_t i = 0; i < key_num; i++) {
      if (host_found[i]) found_num++;
    }
    EXPECT_EQ(found_num, key_num) << "Failed at update iteration " << t;

    free_device_mem(device_values_ptr);
    free_device_mem(device_found);
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试11：清表后 assign_values 测试
TEST_F(AssignValuesTest, assign_after_clear) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  // 插入数据
  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 清表
  table.clear(stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);

  // 清表后执行 assign_values，应该不会影响表（keys 不存在）
  vector<V> new_values(key_num * dim, 9999.0f);
  copy_to_device(device_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试12：乱序更新 values 测试
TEST_F(AssignValuesTest, shuffled_keys_update) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 512;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 打乱 keys 顺序进行更新
  vector<K> shuffled_keys = host_keys;
  vector<V> shuffled_values(key_num * dim);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), g);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < dim; j++) {
      shuffled_values[i * dim + j] =
          static_cast<V>(shuffled_keys[i]) * 0.00001f + 8888.0f;
    }
  }

  copy_to_device(device_keys, shuffled_keys.data(), key_num);
  copy_to_device(device_values, shuffled_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证所有 keys 仍然存在
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) found_num++;
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
}

// 测试13：使用 export_batch 验证 values 更新
TEST_F(AssignValuesTest, verify_with_export_batch) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 更新 values
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < dim; j++) {
      new_values[i * dim + j] = 50000.0f + static_cast<V>(i * dim + j);
    }
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_new_values, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 export_batch 导出并验证
  K* export_keys = alloc_device_mem<K>(key_num);
  V* export_values = alloc_device_mem<V>(key_num * dim);
  S* export_scores = alloc_device_mem<S>(key_num);

  size_t exported = table.export_batch(init_capacity, 0, export_keys,
                                       export_values, export_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(exported, key_num);

  vector<K> real_keys(key_num, 0);
  vector<V> real_values(key_num * dim, 0);
  copy_to_host(real_keys.data(), export_keys, exported);
  copy_to_host(real_values.data(), export_values, exported * dim);

  // 构建 key 到 value 的映射用于验证
  std::unordered_map<K, size_t> key_to_idx;
  for (size_t i = 0; i < key_num; i++) {
    key_to_idx[host_keys[i]] = i;
  }

  // 验证所有原始 keys 都被导出，且 values 已更新
  for (size_t i = 0; i < exported; i++) {
    K k = real_keys[i];
    EXPECT_TRUE(key_to_idx.find(k) != key_to_idx.end())
        << "Key " << k << " not found in original keys";
    size_t orig_idx = key_to_idx[k];
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(real_values[i * dim + j], new_values[orig_idx * dim + j])
          << "Value mismatch for key " << k << " dim " << j;
    }
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(export_keys);
  free_device_mem(export_values);
  free_device_mem(export_scores);
}

// 测试14：unique_key 参数测试
TEST_F(AssignValuesTest, unique_key_param) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 128;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 unique_key=true 调用 assign_values
  vector<V> new_values(key_num * dim, 7777.0f);
  copy_to_device(device_values, new_values.data(), key_num * dim);

  table.assign_values(key_num, device_keys, device_values, stream_, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) found_num++;
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
}

// 测试15：unique_key=false 重复 key 更新
TEST_F(AssignValuesTest, unique_key_false_duplicate_keys) {
  constexpr size_t dim = 8;
  constexpr size_t unique_key_num = 128;
  constexpr size_t assign_key_num = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> unique_keys(unique_key_num, 0);
  vector<V> insert_values(unique_key_num * dim, 0);
  vector<S> insert_scores(unique_key_num, 0);
  create_continuous_keys<K, S, V, dim>(unique_keys.data(), insert_scores.data(),
                                       insert_values.data(), unique_key_num);

  K* device_unique_keys = alloc_device_mem<K>(unique_key_num);
  V* device_insert_values = alloc_device_mem<V>(unique_key_num * dim);
  S* device_insert_scores = alloc_device_mem<S>(unique_key_num);

  copy_to_device(device_unique_keys, unique_keys.data(), unique_key_num);
  copy_to_device(device_insert_values, insert_values.data(),
                 unique_key_num * dim);
  copy_to_device(device_insert_scores, insert_scores.data(), unique_key_num);

  table.insert_or_assign(unique_key_num, device_unique_keys,
                         device_insert_values, device_insert_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), unique_key_num);

  // 构造带重复 key 的 assign 数组：assign_keys[i] = unique_keys[i %
  // unique_key_num]
  vector<K> assign_keys(assign_key_num, 0);
  vector<V> assign_values(assign_key_num * dim, 0);
  for (size_t i = 0; i < assign_key_num; i++) {
    assign_keys[i] = unique_keys[i % unique_key_num];
    for (size_t j = 0; j < dim; j++) {
      assign_values[i * dim + j] = 1000.0f + static_cast<V>(i) + j * 0.1f;
    }
  }

  K* device_assign_keys = alloc_device_mem<K>(assign_key_num);
  V* device_assign_values = alloc_device_mem<V>(assign_key_num * dim);

  copy_to_device(device_assign_keys, assign_keys.data(), assign_key_num);
  copy_to_device(device_assign_values, assign_values.data(),
                 assign_key_num * dim);

  table.assign_values(assign_key_num, device_assign_keys, device_assign_values,
                      stream_, false);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), unique_key_num);

  // 验证：对每个唯一 key 做 find，均应找到；value 应为本次 assign 中该 key
  // 某一次写入
  V** device_values_ptr = alloc_device_mem<V*>(unique_key_num);
  bool* device_found = alloc_device_mem<bool>(unique_key_num);
  V* device_values_buffer = alloc_device_mem<V>(unique_key_num * dim);

  table.find(unique_key_num, device_unique_keys, device_values_ptr,
             device_found, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, unique_key_num,
                stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(unique_key_num);
  copy_to_host(host_found.get(), device_found, unique_key_num);
  vector<V> real_values(unique_key_num * dim, 0);
  copy_to_host(real_values.data(), device_values_buffer, unique_key_num * dim);

  size_t found_count = 0;
  for (size_t i = 0; i < unique_key_num; i++) {
    EXPECT_TRUE(host_found[i])
        << "Unique key at index " << i << " should be found";
    if (host_found[i]) {
      found_count++;
      // 该 key 在 assign 中出现的下标为 i, i+unique_key_num，对应 value 为
      // assign_values[i] 或 assign_values[i+unique_key_num]
      bool match_first = true;
      bool match_second = true;
      for (size_t j = 0; j < dim; j++) {
        V expected_first = assign_values[i * dim + j];
        V expected_second = assign_values[(i + unique_key_num) * dim + j];
        if (std::abs(real_values[i * dim + j] - expected_first) > 1e-5f) {
          match_first = false;
        }
        if (std::abs(real_values[i * dim + j] - expected_second) > 1e-5f) {
          match_second = false;
        }
      }
      EXPECT_TRUE(match_first || match_second)
          << "Value for key at index " << i
          << " should match one of the assigned values";
    }
  }
  EXPECT_EQ(found_count, unique_key_num);

  free_device_mem(device_unique_keys);
  free_device_mem(device_insert_values);
  free_device_mem(device_insert_scores);
  free_device_mem(device_assign_keys);
  free_device_mem(device_assign_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_values_buffer);
}

void test_repeat_key_assign_values_basic(size_t dim, size_t hbm_for_values,
                                         bool io_by_cpu) {
  constexpr size_t key_num = 1UL * 1024;
  constexpr size_t repeat_num = 10;
  constexpr size_t unique_key_num = (key_num + repeat_num - 1) / repeat_num;

  using Table = npu::hkv::HashTable<K, V, S>;
  auto table = std::make_unique<Table>();
  npu::hkv::HashTableOptions options{
      .init_capacity = DEFAULT_CAPACITY,
      .max_capacity = DEFAULT_CAPACITY,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = io_by_cpu,
  };
  table->init(options);
  EXPECT_EQ(table->size(), 0);

  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num, dim);
  vector<K> host_keys(key_num, 0);
  vector<V> host_insert_values(key_num * dim, 0);
  create_continuous_keys<K, S, V>(dim, host_keys.data(), nullptr,
                                  host_insert_values.data(), key_num);
  for (size_t i = 0; i < key_num; i++) {
    host_keys[i] = i / repeat_num;
  }
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_insert_values, key_num, dim);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), unique_key_num);

  vector<V> host_assign_values(key_num * dim, 0);
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < dim; j++) {
      host_assign_values[i * dim + j] =
          1000.0f + static_cast<V>(host_keys[i]) * 0.01f + static_cast<V>(j);
    }
  }
  device_data.copy_values(host_assign_values, key_num, dim);
  table->assign_values(key_num, device_data.device_keys, device_data.device_values,
                       device_data.stream, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  DeviceData<K, V, S> find_data;
  find_data.malloc(key_num, dim);
  table->find(key_num, device_data.device_keys, find_data.device_values_ptr,
              find_data.device_found, nullptr, find_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(find_data.stream), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  ASSERT_EQ(aclrtMemcpy(host_found.get(), key_num * sizeof(bool),
                        find_data.device_found, key_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  vector<V*> found_values(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(found_values.data(), key_num * sizeof(V*),
                        find_data.device_values_ptr, key_num * sizeof(V*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  set<K> checked_keys;
  for (size_t i = 0; i < key_num; i++) {
    if (!host_found[i] || checked_keys.find(host_keys[i]) != checked_keys.end()) {
      continue;
    }
    checked_keys.insert(host_keys[i]);

    vector<V> real_values(dim, 0);
    ASSERT_EQ(aclrtMemcpy(real_values.data(), dim * find_data.each_value_size,
                          found_values[i], dim * find_data.each_value_size,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    bool matched = false;
    size_t search_start = (i / repeat_num) * repeat_num;
    for (size_t j = search_start; j < search_start + repeat_num && j < key_num;
         j++) {
      vector<V> candidate_values(host_assign_values.begin() + j * dim,
                                 host_assign_values.begin() + j * dim + dim);
      if (host_keys[i] == host_keys[j] && candidate_values == real_values) {
        matched = true;
        break;
      }
    }
    ASSERT_TRUE(matched) << "key: " << host_keys[i] << " value mismatch";
  }
  EXPECT_EQ(checked_keys.size(), unique_key_num);

  table->clear(find_data.stream);
  EXPECT_EQ(table->size(find_data.stream), 0);
}

TEST_F(AssignValuesTest, test_repeat_key_with_pure_hbm) {
  constexpr size_t hbm = numeric_limits<size_t>::max();
  test_repeat_key_assign_values_basic(20480, hbm, false);
}

TEST_F(AssignValuesTest, test_repeat_key_with_ddr) {
  constexpr size_t hbm = 4UL << 30;
  test_repeat_key_assign_values_basic(20480, hbm, false);
}

TEST_F(AssignValuesTest, test_repeat_key_with_pure_ddr) {
  test_repeat_key_assign_values_basic(DEFAULT_DIM, 0, false);
}

TEST_F(AssignValuesTest, test_repeat_key_with_io_by_cpu) {
  test_repeat_key_assign_values_basic(DEFAULT_DIM, 0, true);
}

void test_ddr_assign_values_dim_basic(size_t dim, bool io_by_cpu = false) {
  // 1. 建表
  constexpr size_t capacity = 128UL;
  auto table = std::make_unique<npu::hkv::HashTable<K, V, S>>();

  npu::hkv::HashTableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors =
      io_by_cpu ? 0 : capacity * sizeof(V) * dim / 2;  // 一半device一半host
  options.dim = dim;
  options.io_by_cpu = io_by_cpu;

  table->init(options);
  EXPECT_EQ(table->size(), 0);

  // 2. 申请hbm内存
  constexpr size_t key_num = 128;
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num, dim);

  // 3. 空桶插值
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  create_continuous_keys<K, S, V>(dim, host_keys.data(), nullptr,
                                  host_values.data(), key_num, 0);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num, dim);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 4. 使用 assign_values 更新 values
  vector<V> new_values(key_num * dim, 0);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 100.0f;
  }
  device_data.copy_values(new_values, key_num, dim);
  table->assign_values(key_num, device_data.device_keys,
                      device_data.device_values, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 5. 校验更新结果 - 使用 find 接口验证
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  check_result(new_values, key_num, device_data, key_num, dim);
}

TEST_F(AssignValuesTest, test_ddr_dim_8) { test_ddr_assign_values_dim_basic(8); }

TEST_F(AssignValuesTest, test_ddr_dim_1024) {
  test_ddr_assign_values_dim_basic(1024);
}

TEST_F(AssignValuesTest, test_ddr_dim_8_by_cpu) {
  test_ddr_assign_values_dim_basic(8, true);
}

TEST_F(AssignValuesTest, test_ddr_dim_1024_by_cpu) {
  test_ddr_assign_values_dim_basic(1024, true);
}
