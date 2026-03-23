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
#include <memory>
#include <random>
#include <unordered_map>
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
