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
#include <limits>
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
class AssignTest : public ::testing::Test {
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

// 测试1：基本功能测试 - 插入后更新 values 和 scores
TEST_F(AssignTest, basic_function) {
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

  // 更新 values 和 scores 为新值
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 1.0f;
  }
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] + 1000;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  S* device_new_scores = alloc_device_mem<S>(key_num);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);
  copy_to_device(device_new_scores, new_scores.data(), key_num);

  table.assign(key_num, device_keys, device_new_values, device_new_scores,
               stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 find 接口验证 values 和 scores 均已更新
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(key_num * dim);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> real_values(key_num * dim, 0);
  vector<S> real_scores(key_num, 0);
  copy_to_host(real_values.data(), device_values_buffer, key_num * dim);
  copy_to_host(real_scores.data(), device_out_scores, key_num);
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
      EXPECT_NE(real_scores[i], host_scores[i])
          << "Score at index " << i << " should be updated";
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(device_new_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
  free_device_mem(device_values_buffer);
}

// 测试2：空表测试 - 对空表执行 assign 不会崩溃
TEST_F(AssignTest, empty_table) {
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
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  // 在空表上执行 assign 应该不会崩溃
  table.assign(key_num, device_keys, device_values, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试3：边界情况 - n=0 时不崩溃
TEST_F(AssignTest, zero_keys) {
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

  // n=0 时调用 assign 应该直接返回，不崩溃
  table.assign(0, nullptr, nullptr, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);
}

// 测试4：单个 key 测试 - n=1
TEST_F(AssignTest, single_key) {
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

  // 更新 value 和 score
  vector<V> new_values(dim, 9.9f);
  S new_score = 99999;
  copy_to_device(device_values, new_values.data(), dim);
  copy_to_device(device_scores, &new_score, key_num);

  table.assign(key_num, device_keys, device_values, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证 value 和 score 均已更新
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(dim);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
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

  S real_score = 0;
  copy_to_host(&real_score, device_out_scores, 1);
  EXPECT_NE(real_score, host_score) << "Score should be updated";

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
  free_device_mem(device_values_buffer);
}

// 测试5：部分 keys 存在测试 - 只更新存在的 keys 的 values 和 scores
TEST_F(AssignTest, partial_keys_exist) {
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
  vector<S> update_scores(update_key_num, 0);
  create_continuous_keys<K, S, V, dim>(update_keys.data(), update_scores.data(),
                                       update_values.data(), update_key_num, 1);

  for (size_t i = 0; i < update_key_num * dim; i++) {
    update_values[i] = 100.0f + static_cast<V>(i % dim);
  }
  for (size_t i = 0; i < update_key_num; i++) {
    update_scores[i] = 5000 + i;
  }

  K* device_update_keys = alloc_device_mem<K>(update_key_num);
  V* device_update_values = alloc_device_mem<V>(update_key_num * dim);
  S* device_update_scores = alloc_device_mem<S>(update_key_num);

  copy_to_device(device_update_keys, update_keys.data(), update_key_num);
  copy_to_device(device_update_values, update_values.data(),
                 update_key_num * dim);
  copy_to_device(device_update_scores, update_scores.data(), update_key_num);

  table.assign(update_key_num, device_update_keys, device_update_values,
               device_update_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), insert_key_num);

  // 验证存在的 keys 的 values 和 scores 已更新
  V** device_values_ptr = alloc_device_mem<V*>(insert_key_num);
  bool* device_found = alloc_device_mem<bool>(insert_key_num);
  S* device_out_scores = alloc_device_mem<S>(insert_key_num);
  V* device_values_buffer = alloc_device_mem<V>(insert_key_num * dim);

  table.find(insert_key_num, device_insert_keys, device_values_ptr,
             device_found, device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, insert_key_num,
                stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(insert_key_num);
  copy_to_host(host_found.get(), device_found, insert_key_num);
  vector<V> real_values(insert_key_num * dim, 0);
  vector<S> real_scores(insert_key_num, 0);
  copy_to_host(real_values.data(), device_values_buffer, insert_key_num * dim);
  copy_to_host(real_scores.data(), device_out_scores, insert_key_num);

  size_t found_count = 0;
  for (size_t i = 0; i < insert_key_num; i++) {
    if (host_found[i]) {
      found_count++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(real_values[i * dim + j], update_values[i * dim + j])
            << "Value at index " << i << " dim " << j << " should be updated";
      }
      EXPECT_NE(real_scores[i], insert_scores[i])
          << "Score at index " << i << " should be updated";
    }
  }
  EXPECT_EQ(found_count, insert_key_num);

  free_device_mem(device_insert_keys);
  free_device_mem(device_insert_values);
  free_device_mem(device_insert_scores);
  free_device_mem(device_update_keys);
  free_device_mem(device_update_values);
  free_device_mem(device_update_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
  free_device_mem(device_values_buffer);
}

// 测试6：大规模数据测试
TEST_F(AssignTest, large_scale) {
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

  // 更新 values 和 scores
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 10.0f;
  }
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] + 10000;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  S* device_new_scores = alloc_device_mem<S>(key_num);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);
  copy_to_device(device_new_scores, new_scores.data(), key_num);

  table.assign(key_num, device_keys, device_new_values, device_new_scores,
               stream_);
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
  free_device_mem(device_new_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
}

// 测试7：不同 dim 测试 - dim=128
TEST_F(AssignTest, large_dim) {
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

  // 更新 values 和 scores
  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] * 3.0f;
  }
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] * 3;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  S* device_new_scores = alloc_device_mem<S>(key_num);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);
  copy_to_device(device_new_scores, new_scores.data(), key_num);

  table.assign(key_num, device_keys, device_new_values, device_new_scores,
               stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(key_num * dim);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_values_ptr, device_values_buffer, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);
  vector<V> real_values(key_num * dim, 0);
  vector<S> real_scores(key_num, 0);
  copy_to_host(real_values.data(), device_values_buffer, key_num * dim);
  copy_to_host(real_scores.data(), device_out_scores, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(real_values[i * dim + j], new_values[i * dim + j])
            << "Value at index " << i << " dim " << j << " should be updated";
      }
      EXPECT_NE(real_scores[i], host_scores[i])
          << "Score at index " << i << " should be updated";
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_new_values);
  free_device_mem(device_new_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
  free_device_mem(device_values_buffer);
}

// 测试8：多次更新 values 和 scores 测试
TEST_F(AssignTest, multiple_updates) {
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

  for (size_t t = 0; t < update_times; t++) {
    vector<V> new_values(key_num * dim);
    for (size_t i = 0; i < key_num * dim; i++) {
      new_values[i] = static_cast<V>((t + 1) * 1000 + (i % dim));
    }
    vector<S> new_scores(key_num);
    for (size_t i = 0; i < key_num; i++) {
      new_scores[i] = (t + 1) * 1000 + i;
    }
    copy_to_device(device_values, new_values.data(), key_num * dim);
    copy_to_device(device_scores, new_scores.data(), key_num);

    table.assign(key_num, device_keys, device_values, device_scores, stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

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

// 测试9：乱序更新 values 和 scores 测试
TEST_F(AssignTest, shuffled_keys_update) {
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

  vector<K> shuffled_keys = host_keys;
  vector<V> shuffled_values(key_num * dim);
  vector<S> shuffled_scores(key_num);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), g);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < dim; j++) {
      shuffled_values[i * dim + j] =
          static_cast<V>(shuffled_keys[i]) * 0.00001f + 8888.0f;
    }
    shuffled_scores[i] = shuffled_keys[i] + 8888;
  }

  copy_to_device(device_keys, shuffled_keys.data(), key_num);
  copy_to_device(device_values, shuffled_values.data(), key_num * dim);
  copy_to_device(device_scores, shuffled_scores.data(), key_num);

  table.assign(key_num, device_keys, device_values, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

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

// 测试10：unique_key=false，批次内 key 互不相同（走 assign_kernel_with_io），
// 更新结果应与 unique_key=true 时一致
TEST_F(AssignTest, unique_key_false_distinct_keys_updates_correctly) {
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
  EXPECT_EQ(table.size(), key_num);

  vector<V> new_values(key_num * dim);
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 42.0f;
  }
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] + 5000;
  }
  copy_to_device(device_values, new_values.data(), key_num * dim);
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign(key_num, device_keys, device_values, device_scores, stream_,
               false);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);
  V* device_values_buffer = alloc_device_mem<V>(key_num * dim);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  read_from_ptr(device_values_ptr, device_values_buffer, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> real_values(key_num * dim, 0);
  vector<S> real_scores(key_num, 0);
  copy_to_host(real_values.data(), device_values_buffer, key_num * dim);
  copy_to_host(real_scores.data(), device_out_scores, key_num);
  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    ASSERT_TRUE(host_found[i]) << "key index " << i;
    found_num++;
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(real_values[i * dim + j], new_values[i * dim + j])
          << "i=" << i << " j=" << j;
    }
    // kLfu：assign_kernel_with_io 将 score 写为 原分数 + 输入分数
    EXPECT_EQ(real_scores[i], host_scores[i] + new_scores[i])
        << "score i=" << i;
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
  free_device_mem(device_values_buffer);
}

// 测试11：unique_key=false，批次内同一 key 出现多次且 value/score 完全相同，
// 应稳定更新为该公共向量与分数（避免不同 payload 下的写入竞态）
TEST_F(AssignTest, unique_key_false_duplicate_keys_same_payload) {
  constexpr size_t dim = 8;
  constexpr size_t n_dup = 96;
  constexpr K dup_key = 900001ULL;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> ins_k = {dup_key};
  vector<V> ins_v(dim);
  vector<S> ins_s = {1000};
  for (size_t j = 0; j < dim; j++) {
    ins_v[j] = static_cast<V>(j);
  }

  K* d_ins_k = alloc_device_mem<K>(1);
  V* d_ins_v = alloc_device_mem<V>(dim);
  S* d_ins_s = alloc_device_mem<S>(1);
  copy_to_device(d_ins_k, ins_k.data(), 1);
  copy_to_device(d_ins_v, ins_v.data(), dim);
  copy_to_device(d_ins_s, ins_s.data(), 1);
  table.insert_or_assign(1, d_ins_k, d_ins_v, d_ins_s, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 1);

  vector<K> host_keys(n_dup, dup_key);
  vector<V> host_vals(n_dup * dim);
  vector<S> host_scores(n_dup, 8888);
  const V fill = 3.25f;
  for (size_t i = 0; i < n_dup; i++) {
    for (size_t j = 0; j < dim; j++) {
      host_vals[i * dim + j] = fill + static_cast<V>(j) * 0.01f;
    }
  }

  K* d_keys = alloc_device_mem<K>(n_dup);
  V* d_vals = alloc_device_mem<V>(n_dup * dim);
  S* d_scores = alloc_device_mem<S>(n_dup);
  copy_to_device(d_keys, host_keys.data(), n_dup);
  copy_to_device(d_vals, host_vals.data(), n_dup * dim);
  copy_to_device(d_scores, host_scores.data(), n_dup);

  table.assign(n_dup, d_keys, d_vals, d_scores, stream_, false);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  V** d_vptr = alloc_device_mem<V*>(1);
  bool* d_found = alloc_device_mem<bool>(1);
  S* d_out_s = alloc_device_mem<S>(1);
  table.find(1, d_ins_k, d_vptr, d_found, d_out_s, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  V* d_buf = alloc_device_mem<V>(dim);
  read_from_ptr(d_vptr, d_buf, dim, 1, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> got(dim);
  bool hf = false;
  S got_s = 0;
  copy_to_host(got.data(), d_buf, dim);
  copy_to_host(&hf, d_found, 1);
  copy_to_host(&got_s, d_out_s, 1);
  ASSERT_TRUE(hf);
  for (size_t j = 0; j < dim; j++) {
    EXPECT_FLOAT_EQ(got[j], host_vals[0 * dim + j]) << "dim " << j;
  }
  // 同一 key 多线程更新分数存在竞态，仅校验分数已按 LFU 语义增长
  EXPECT_GE(got_s, ins_s[0]);

  free_device_mem(d_ins_k);
  free_device_mem(d_ins_v);
  free_device_mem(d_ins_s);
  free_device_mem(d_keys);
  free_device_mem(d_vals);
  free_device_mem(d_scores);
  free_device_mem(d_vptr);
  free_device_mem(d_found);
  free_device_mem(d_out_s);
  free_device_mem(d_buf);
}

// 测试12：kLru + unique_key=false + scores=nullptr，应对互异 key 正确更新向量
TEST_F(AssignTest, unique_key_false_lru_scores_nullptr) {
  constexpr size_t dim = 4;
  constexpr size_t key_num = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLru> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);

  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> new_values(key_num * dim);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 55.0f;
  }
  V* device_new_values = alloc_device_mem<V>(key_num * dim);
  copy_to_device(device_new_values, new_values.data(), key_num * dim);

  table.assign(key_num, device_keys, device_new_values, nullptr, stream_,
               false);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  V* device_buf = alloc_device_mem<V>(key_num * dim);
  read_from_ptr(device_values_ptr, device_buf, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> real(key_num * dim);
  copy_to_host(real.data(), device_buf, key_num * dim);
  auto host_found = std::make_unique<bool[]>(key_num);
  copy_to_host(host_found.get(), device_found, key_num);

  for (size_t i = 0; i < key_num; i++) {
    ASSERT_TRUE(host_found[i]);
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(real[i * dim + j], new_values[i * dim + j]);
    }
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_new_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_buf);
}

void test_repeat_key_assign_basic(size_t dim, size_t hbm_for_values,
                                  bool io_by_cpu) {
  constexpr size_t key_num = 1UL * 1024;
  constexpr size_t repeat_num = 10;
  constexpr size_t unique_key_num = (key_num + repeat_num - 1) / repeat_num;

  using LocalKey = int64_t;
  using Table = npu::hkv::HashTable<LocalKey, V, S, EvictStrategy::kLfu>;
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

  DeviceData<LocalKey, V, S> device_data;
  device_data.malloc(key_num, dim);

  vector<LocalKey> host_keys(key_num, 0);
  vector<V> host_insert_values(key_num * dim, 0);
  vector<S> host_insert_scores(key_num, 0);
  create_continuous_keys<LocalKey, S, V>(dim, host_keys.data(),
                                         host_insert_scores.data(),
                                         host_insert_values.data(), key_num);
  for (size_t i = 0; i < key_num; i++) {
    host_keys[i] = static_cast<LocalKey>(i / repeat_num);
    host_insert_scores[i] = static_cast<S>(100 + i);
  }
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_insert_values, key_num, dim);
  device_data.copy_scores(host_insert_scores, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), unique_key_num);

  vector<V> host_assign_values(key_num * dim, 0);
  vector<S> host_assign_scores(key_num, 0);
  for (size_t i = 0; i < key_num; i++) {
    host_assign_scores[i] = static_cast<S>(5000 + i);
    for (size_t j = 0; j < dim; j++) {
      host_assign_values[i * dim + j] = 2000.0f + static_cast<V>(host_keys[i]) +
                                        static_cast<V>(j) * 0.1f;
    }
  }
  device_data.copy_values(host_assign_values, key_num, dim);
  device_data.copy_scores(host_assign_scores, key_num);
  table->assign(key_num, device_data.device_keys, device_data.device_values,
                device_data.device_scores, device_data.stream, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  DeviceData<LocalKey, V, S> find_data;
  find_data.malloc(key_num, dim);
  table->find(key_num, device_data.device_keys, find_data.device_values_ptr,
              find_data.device_found, find_data.device_scores, find_data.stream);
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
  vector<S> found_scores(key_num, 0);
  ASSERT_EQ(aclrtMemcpy(found_scores.data(), key_num * sizeof(S),
                        find_data.device_scores, key_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  set<LocalKey> checked_keys;
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

    bool value_matched = false;
    size_t search_start = (i / repeat_num) * repeat_num;
    S score_floor = std::numeric_limits<S>::max();
    for (size_t j = search_start; j < search_start + repeat_num && j < key_num;
         j++) {
      if (host_keys[j] != host_keys[i]) {
        continue;
      }
      vector<V> candidate_values(host_assign_values.begin() + j * dim,
                                 host_assign_values.begin() + j * dim + dim);
      if (candidate_values == real_values) {
        value_matched = true;
      }
      score_floor = std::min(score_floor, host_insert_scores[j]);
    }
    ASSERT_TRUE(value_matched) << "key: " << host_keys[i] << " value mismatch";
    EXPECT_GE(found_scores[i], score_floor)
        << "key: " << host_keys[i] << " score should not decrease";
  }
  EXPECT_EQ(checked_keys.size(), unique_key_num);

  table->clear(find_data.stream);
  EXPECT_EQ(table->size(find_data.stream), 0);
}

TEST_F(AssignTest, test_repeat_key_with_pure_hbm) {
  constexpr size_t hbm = numeric_limits<size_t>::max();
  test_repeat_key_assign_basic(20480, hbm, false);
}

TEST_F(AssignTest, test_repeat_key_with_ddr) {
  constexpr size_t hbm = 4UL << 30;
  test_repeat_key_assign_basic(20480, hbm, false);
}

TEST_F(AssignTest, test_repeat_key_with_pure_ddr) {
  test_repeat_key_assign_basic(DEFAULT_DIM, 0, false);
}

TEST_F(AssignTest, test_repeat_key_with_io_by_cpu) {
  test_repeat_key_assign_basic(DEFAULT_DIM, 0, true);
}

void test_ddr_assign_dim_basic(size_t dim, bool io_by_cpu = false) {
  // 1. 建表
  constexpr size_t capacity = 128UL;
  auto table = std::make_unique<
      npu::hkv::HashTable<K, V, S, EvictStrategy::kCustomized>>();

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
  vector<S> host_scores(key_num, 0);

  create_continuous_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                                  host_values.data(), key_num, 0);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num, dim);
  device_data.copy_scores(host_scores, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 4. 使用 assign_values 更新 values
  vector<V> new_values(key_num * dim, 0);
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num * dim; i++) {
    new_values[i] = host_values[i] + 100.0f;
  }
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] + 5000;
  }
  device_data.copy_values(new_values, key_num, dim);
  device_data.copy_scores(new_scores, key_num);

  table->assign(key_num, device_data.device_keys, device_data.device_values,
                device_data.device_scores, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 5. 校验更新结果 - 使用 find 接口验证
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, device_data.device_scores,
              device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  vector<V> host_result_values(key_num * dim);
  bool* host_result_found = nullptr;
  vector<S> host_result_scores(key_num);

  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_result_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);

  ASSERT_EQ(aclrtMemcpy(host_result_values.data(), key_num * dim * sizeof(V),
                        device_data.device_values, key_num * dim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_result_found, key_num * sizeof(bool),
                        device_data.device_found, key_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_result_scores.data(), key_num * sizeof(S),
                        device_data.device_scores, key_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(host_result_found[i])
        << "Key " << host_keys[i] << " should be found";
    for (size_t j = 0; j < dim; j++) {
      EXPECT_EQ(host_result_values[i * dim + j], new_values[i * dim + j])
          << "Value mismatch for key " << host_keys[i] << " at dim " << j;
    }
    EXPECT_EQ(host_result_scores[i], new_scores[i])
        << "Score mismatch for key " << host_keys[i] << ": expected "
        << new_scores[i] << ", got " << host_result_scores[i];
  }

  ASSERT_EQ(aclrtFreeHost(host_result_found), ACL_ERROR_NONE);

  table->clear();
}

TEST_F(AssignTest, test_ddr_dim_8) { test_ddr_assign_dim_basic(8); }

TEST_F(AssignTest, test_ddr_dim_1024) { test_ddr_assign_dim_basic(1024); }

TEST_F(AssignTest, test_ddr_dim_8_by_cpu) {
  test_ddr_assign_dim_basic(8, true);
}

TEST_F(AssignTest, test_ddr_dim_1024_by_cpu) {
  test_ddr_assign_dim_basic(1024, true);
}
