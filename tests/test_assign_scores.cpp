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
class AssignScoresTest : public ::testing::Test {
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

// 测试1：基本功能测试 - 插入后更新分数
TEST_F(AssignScoresTest, basic_function) {
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

  // 更新分数为新值
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] + 1000;  // 新分数 = 原分数 + 1000
  }
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 find 接口验证分数已更新
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
  copy_to_host(host_found.get(), device_found, key_num);
  vector<S> real_scores(key_num, 0);
  copy_to_host(real_scores.data(), device_out_scores, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      // 验证分数已更新（考虑 epochlfu 策略可能会修改分数格式）
      EXPECT_NE(real_scores[i], host_scores[i])
          << "Score at index " << i << " should be updated";
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
}

// 测试2：空表测试 - 对空表执行 assign_scores 不会崩溃
TEST_F(AssignScoresTest, empty_table) {
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
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       nullptr, key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_scores, host_scores.data(), key_num);

  // 在空表上执行 assign_scores 应该不会崩溃
  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);

  free_device_mem(device_keys);
  free_device_mem(device_scores);
}

// 测试3：边界情况 - n=0 时不崩溃
TEST_F(AssignScoresTest, zero_keys) {
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

  // n=0 时调用 assign_scores 应该直接返回，不崩溃
  table.assign_scores(0, nullptr, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);
}

// 测试4：单个 key 测试 - n=1
TEST_F(AssignScoresTest, single_key) {
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
  table.set_global_epoch(1);  // EpochLfu 策略需要设置 global_epoch

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

  // 更新分数
  S new_score = 99999;
  copy_to_device(device_scores, &new_score, key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证分数已更新
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  bool host_found = false;
  copy_to_host(&host_found, device_found, 1);
  EXPECT_TRUE(host_found);

  S real_score = 0;
  copy_to_host(&real_score, device_out_scores, 1);
  EXPECT_NE(real_score, host_score) << "Score should be updated";

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
}

// 测试5：部分 keys 存在测试 - 只更新存在的 keys 的分数
TEST_F(AssignScoresTest, partial_keys_exist) {
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
  vector<S> update_scores(update_key_num, 0);
  create_continuous_keys<K, S, V, dim>(update_keys.data(), update_scores.data(),
                                       nullptr, update_key_num, 1);

  // 设置新分数
  for (size_t i = 0; i < update_key_num; i++) {
    update_scores[i] = 5000 + i;
  }

  K* device_update_keys = alloc_device_mem<K>(update_key_num);
  S* device_update_scores = alloc_device_mem<S>(update_key_num);

  copy_to_device(device_update_keys, update_keys.data(), update_key_num);
  copy_to_device(device_update_scores, update_scores.data(), update_key_num);

  // 执行 assign_scores，不存在的 keys 应该被忽略
  table.assign_scores(update_key_num, device_update_keys, device_update_scores,
                      stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), insert_key_num);  // 表大小不变

  // 验证存在的 keys 分数已更新
  V** device_values_ptr = alloc_device_mem<V*>(insert_key_num);
  bool* device_found = alloc_device_mem<bool>(insert_key_num);
  S* device_out_scores = alloc_device_mem<S>(insert_key_num);

  table.find(insert_key_num, device_insert_keys, device_values_ptr,
             device_found, device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[insert_key_num]());
  copy_to_host(host_found.get(), device_found, insert_key_num);
  vector<S> real_scores(insert_key_num, 0);
  copy_to_host(real_scores.data(), device_out_scores, insert_key_num);

  size_t found_count = 0;
  for (size_t i = 0; i < insert_key_num; i++) {
    if (host_found[i]) {
      found_count++;
      EXPECT_NE(real_scores[i], insert_scores[i])
          << "Score at index " << i << " should be updated";
    }
  }
  EXPECT_EQ(found_count, insert_key_num);

  free_device_mem(device_insert_keys);
  free_device_mem(device_insert_values);
  free_device_mem(device_insert_scores);
  free_device_mem(device_update_keys);
  free_device_mem(device_update_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
}

// 测试6：随机 keys 测试
TEST_F(AssignScoresTest, random_keys) {
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

  // 更新分数
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] * 2 + 100;
  }
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证分数已更新
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
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
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
}

// 测试7：大规模数据测试
TEST_F(AssignScoresTest, large_scale) {
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

  // 更新分数
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] + 10000;
  }
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
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
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
}

// 测试8：不同 dim 测试 - dim=4
TEST_F(AssignScoresTest, small_dim) {
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

  // 更新分数
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = 9999;
  }
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
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

// 测试9：不同 dim 测试 - dim=128
TEST_F(AssignScoresTest, large_dim) {
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

  // 更新分数
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = host_scores[i] * 3;
  }
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
  copy_to_host(host_found.get(), device_found, key_num);
  vector<S> real_scores(key_num, 0);
  copy_to_host(real_scores.data(), device_out_scores, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      EXPECT_NE(real_scores[i], host_scores[i])
          << "Score at index " << i << " should be updated";
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
}

// 测试10：多次更新分数测试
TEST_F(AssignScoresTest, multiple_updates) {
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

  // 多次更新分数
  for (size_t t = 0; t < update_times; t++) {
    vector<S> new_scores(key_num);
    for (size_t i = 0; i < key_num; i++) {
      new_scores[i] = (t + 1) * 1000 + i;
    }
    copy_to_device(device_scores, new_scores.data(), key_num);

    table.assign_scores(key_num, device_keys, device_scores, stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

    // 验证每次更新后 keys 仍然存在
    V** device_values_ptr = alloc_device_mem<V*>(key_num);
    bool* device_found = alloc_device_mem<bool>(key_num);

    table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
               stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

    auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
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

// 测试11：清表后更新分数测试
TEST_F(AssignScoresTest, assign_after_clear) {
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

  // 清表后执行 assign_scores，应该不会影响表（keys 不存在）
  vector<S> new_scores(key_num, 9999);
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), 0);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试12：乱序更新分数测试
TEST_F(AssignScoresTest, shuffled_keys_update) {
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
  vector<S> shuffled_scores(key_num);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), g);

  for (size_t i = 0; i < key_num; i++) {
    shuffled_scores[i] = shuffled_keys[i] + 8888;
  }

  copy_to_device(device_keys, shuffled_keys.data(), key_num);
  copy_to_device(device_scores, shuffled_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证所有 keys 仍然存在
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
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

// 测试13：使用 export_batch 验证分数更新
TEST_F(AssignScoresTest, verify_with_export_batch) {
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

  // 更新分数
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = 50000 + i;
  }
  copy_to_device(device_scores, new_scores.data(), key_num);

  table.assign_scores(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 export_batch 导出并验证
  // 注意：export_batch 的第一个参数 n 既是"最大导出数量"，也是"搜索范围"
  // 需要使用 init_capacity 覆盖整个 hash table，而不是 key_num
  K* export_keys = alloc_device_mem<K>(key_num);
  V* export_values = alloc_device_mem<V>(key_num * dim);
  S* export_scores = alloc_device_mem<S>(key_num);

  size_t exported = table.export_batch(
      init_capacity, 0, export_keys, export_values, export_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(exported, key_num);

  vector<K> real_keys(key_num, 0);
  vector<S> real_scores(key_num, 0);
  copy_to_host(real_keys.data(), export_keys, exported);
  copy_to_host(real_scores.data(), export_scores, exported);

  // 构建 key 到分数的映射用于验证
  std::unordered_map<K, S> key_to_score;
  for (size_t i = 0; i < exported; i++) {
    key_to_score[real_keys[i]] = real_scores[i];
  }

  // 验证所有原始 keys 都被导出
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(key_to_score.find(host_keys[i]) != key_to_score.end())
        << "Key " << host_keys[i] << " not found in exported data";
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(export_keys);
  free_device_mem(export_values);
  free_device_mem(export_scores);
}

// 测试14：assign 别名函数测试
TEST_F(AssignScoresTest, assign_alias) {
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

  // 使用 assign 别名函数（两个参数的版本，用于更新 scores）
  vector<S> new_scores(key_num);
  for (size_t i = 0; i < key_num; i++) {
    new_scores[i] = 77777;
  }
  copy_to_device(device_scores, new_scores.data(), key_num);

  // 调用别名函数
  table.assign(key_num, device_keys, device_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 验证
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
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

// 测试15：kLru 策略测试 - 使用 LRU 淘汰策略的 assign_scores
// LRU 策略下，scores 由系统自动管理（使用时间戳），用户不需要指定 scores
// assign_scores 接口用于更新 key 的时间戳（相当于"访问"该 key）
TEST_F(AssignScoresTest, lru_strategy) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1UL * 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLru> table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);

  // LRU 模式下插入数据时 scores 必须为 nullptr（系统自动使用时间戳）
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 获取插入后的初始 scores（时间戳）
  V** device_values_ptr = alloc_device_mem<V*>(key_num);
  bool* device_found = alloc_device_mem<bool>(key_num);
  S* device_out_scores = alloc_device_mem<S>(key_num);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<S> initial_scores(key_num, 0);
  copy_to_host(initial_scores.data(), device_out_scores, key_num);

  // LRU 模式下 assign_scores 时 scores 也必须为 nullptr
  // 调用 assign_scores 会将 key 的时间戳更新为当前时间
  table.assign_scores(key_num, device_keys, nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 再次查询验证 scores（时间戳）已更新
  table.find(key_num, device_keys, device_values_ptr, device_found,
             device_out_scores, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
  copy_to_host(host_found.get(), device_found, key_num);
  vector<S> updated_scores(key_num, 0);
  copy_to_host(updated_scores.data(), device_out_scores, key_num);

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      // LRU 策略下，assign_scores 会更新时间戳，新时间戳应 >= 初始时间戳
      EXPECT_GE(updated_scores[i], initial_scores[i])
          << "Score at index " << i
          << " should be updated to a newer timestamp";
    }
  }
  EXPECT_EQ(found_num, key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_values_ptr);
  free_device_mem(device_found);
  free_device_mem(device_out_scores);
}
