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
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_template_func.h"
#include "test_util.h"
#include "types.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

template <typename K, typename V, typename S>
void test_erase_basic() {
  // 1. 初始化
  init_env();
  constexpr size_t key_num = 1UL * 1024;

  // 2. 建表
  auto table = get_default_table<K, V, S>();
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 空桶插值
  vector<K> host_keys(key_num, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               nullptr, key_num);
  device_data.copy_keys(host_keys, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 5. 删除部分key
  size_t erase_num = key_num / 2;
  table->erase(erase_num, device_data.device_keys, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num - erase_num);

  // 7. 再插入相同的key，确保size不变
  table->insert_or_assign(erase_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);
}

TEST(TestErase, test_erase_basic) {
  test_erase_basic<uint64_t, float, uint64_t>();
  test_erase_basic<uint64_t, double, uint64_t>();
}

TEST(TestErase, test_erase_in_full_bucket) {
  using K = int64_t;
  using V = float;
  using S = uint64_t;

  // 1. 初始化
  init_env();
  constexpr size_t key_num = 1UL * 1024;

  // 2. 建表，使用capacity为128是为了让1024个key插入后填满整个桶
  constexpr size_t capacity = 128;
  auto table = get_table<K, V, S>(DEFAULT_DIM, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 空桶插值，确保最后一个值在桶内
  vector<K> host_keys(key_num, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               nullptr, key_num);
  device_data.copy_keys(host_keys, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  table->insert_or_assign(1, device_data.device_keys + key_num - 1,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity);

  // 5. 记录当前表内key的存在情况
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMemcpy(host_found, key_num * sizeof(bool), device_data.device_found,
                  key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMemcpy(host_found, key_num * sizeof(bool), device_data.device_found,
                  sizeof(bool) * key_num, ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);

  // 6. 在满桶中删除最后一个key
  table->erase(1, device_data.device_keys + key_num - 1, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity - 1);

  // 7. 再插入相同的key，确保size不变，find结果不变
  table->insert_or_assign(1, device_data.device_keys + key_num - 1,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  bool* host_found_again = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found_again),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found_again, key_num * sizeof(bool),
                        device_data.device_found, sizeof(bool) * key_num,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_EQ(host_found_again[i], host_found[i]);
  }
  aclrtFree(host_found);
  aclrtFree(host_found_again);
}

TEST(TestErase, test_erase_with_rehash) {
  using K = uint64_t;
  using V = float;
  using S = uint64_t;

  // 1. 初始化
  init_env();
  constexpr size_t key_num = 1UL * 1024;
  constexpr size_t init_capacity = 1024;
  constexpr size_t max_capacity = 4096;

  // 2. 建表 - 使用构造函数和init函数，设置更大的max_capacity
  auto table = std::make_unique<npu::hkv::HashTable<K, V, S>>();
  npu::hkv::HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = 4UL << 30,
      .dim = DEFAULT_DIM,
      .num_of_buckets_per_alloc = 1,
  };
  table->init(options);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 插入key
  vector<K> host_keys(key_num, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               nullptr, key_num);
  device_data.copy_keys(host_keys, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 5. 删除部分key
  size_t erase_num = key_num / 2;
  table->erase(erase_num, device_data.device_keys, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num - erase_num);

  // 6. 执行rehash（通过reserve）
  size_t new_capacity = key_num * 2;
  table->reserve(new_capacity, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 7. 验证rehash后表的状态
  EXPECT_EQ(table->size(device_data.stream), key_num - erase_num);
  EXPECT_GE(table->capacity(), new_capacity);

  // 8. 验证剩余的key仍然存在
  table->find(key_num - erase_num, device_data.device_keys + erase_num,
              device_data.device_values_ptr, device_data.device_found, nullptr,
              device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            (key_num - erase_num) * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, (key_num - erase_num) * sizeof(bool),
                        device_data.device_found,
                        (key_num - erase_num) * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  for (size_t i = 0; i < key_num - erase_num; ++i) {
    EXPECT_TRUE(host_found[i]);
  }
  aclrtFree(host_found);
}

TEST(TestErase, test_erase_with_find_or_insert) {
  using K = uint64_t;
  using V = float;
  using S = uint64_t;

  // 1. 初始化
  init_env();
  constexpr size_t key_num = 100;

  // 2. 建表
  auto table = get_default_table<K, V, S>();
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 插入key
  vector<K> host_keys(key_num, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               nullptr, key_num);
  device_data.copy_keys(host_keys, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 5. 删除部分key
  size_t erase_num = key_num / 2;
  table->erase(erase_num, device_data.device_keys, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num - erase_num);

  // 6. 对已删除的key调用find_or_insert（使用device_values_ptr版本）
  table->find_or_insert(erase_num, device_data.device_keys,
                        device_data.device_values_ptr, device_data.device_found,
                        nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 7. 验证表的大小恢复到原始大小
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 8. 验证所有key都存在
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMemcpy(host_found, key_num * sizeof(bool), device_data.device_found,
                  key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);

  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_TRUE(host_found[i]);
  }
  aclrtFree(host_found);
}

TEST(TestErase, test_erase_with_insert_and_evict) {
  using K = uint64_t;
  using V = float;
  using S = uint64_t;

  // 1. 初始化
  init_env();
  constexpr size_t key_num = 128;

  // 2. 建表，使用capacity为128
  constexpr size_t capacity = 128;
  auto table = get_table<K, V, S>(DEFAULT_DIM, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 插入key，填满表
  vector<K> host_keys(key_num, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               nullptr, key_num);
  device_data.copy_keys(host_keys, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity);

  // 5. 删除部分key（这些key会被标记为RECLAIM_KEY）
  size_t erase_num = 10;
  table->erase(erase_num, device_data.device_keys, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity - erase_num);

  // 7. 申请内存用于存储被驱逐的key
  DeviceData<K, V, S> evicted_data;
  evicted_data.malloc(erase_num);
  size_t* d_evicted_counter = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_evicted_counter),
                        sizeof(size_t), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemsetAsync(d_evicted_counter, sizeof(size_t), 0,
                             sizeof(size_t), device_data.stream),
            ACL_ERROR_NONE);

  // 8. 调用insert_and_evict插入新key，应该会驱逐之前标记为RECLAIM_KEY的key
  table->insert_and_evict(
      erase_num, device_data.device_keys, device_data.device_values, nullptr,
      evicted_data.device_keys, evicted_data.device_values,
      evicted_data.device_scores, d_evicted_counter, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 10. 验证驱逐计数为0（因为删除的key会被标记为RECLAIM_KEY，被驱逐后不会返回）
  size_t h_evicted_counter = 0;
  ASSERT_EQ(aclrtMemcpy(&h_evicted_counter, sizeof(size_t), d_evicted_counter,
                        sizeof(size_t), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  EXPECT_EQ(h_evicted_counter, 0);

  // 11. 验证表的大小回到capacity
  EXPECT_EQ(table->size(device_data.stream), capacity);

  // 12. 验证新key存在
  table->find(erase_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            erase_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, erase_num * sizeof(bool),
                        device_data.device_found, erase_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  for (size_t i = 0; i < erase_num; ++i) {
    EXPECT_TRUE(host_found[i]);
  }
  aclrtFree(host_found);
  aclrtFree(d_evicted_counter);
}

enum class EraseIfVersion {
  V1,
  V2
};

template <EraseIfVersion EraseType, typename K, typename V, typename S>
void test_erase_if_basic() {
  // 1. 初始化
  init_env();
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 4 * 1024UL;
  constexpr size_t capacity = 128 * 1024UL;

  // 2. 创建表
  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 创建数据并插入
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);

  // 生成连续的key和scores，其中scores部分大于阈值
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  // 5. 设置测试参数
  constexpr K pattern = 100;
  constexpr S threshold = 0;

  // 计算预期删除的数量
  size_t expected_erase_count = 0;
  for (size_t i = 0; i < key_num; ++i) {
    if (((host_keys[i] & 0x7f) > pattern) && (host_scores[i] > threshold)) {
      expected_erase_count++;
    }
  }
  ASSERT_NE(expected_erase_count, 0);

  // 6. 将数据复制到设备并插入
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  device_data.copy_scores(host_scores, key_num);

  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 7. 根据模板参数选择使用erase_if还是erase_if_v2
  size_t actual_erase_count = 0;
  if constexpr (EraseType == EraseIfVersion::V2) {
    EraseIfPredFunctorV2<K, V, S> pred(pattern, threshold);
    actual_erase_count = table->erase_if_v2(pred, device_data.stream);
  } else {
    actual_erase_count = table->template erase_if<EraseIfPredFunctor>(
        pattern, threshold, device_data.stream);
  }
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 8. 验证删除数量是否符合预期
  EXPECT_EQ(actual_erase_count, expected_erase_count);
  EXPECT_EQ(table->size(device_data.stream), key_num - expected_erase_count);

  // 9. 验证剩余元素是否符合预期
  // 找到应该保留的元素
  vector<K> remaining_keys;
  for (size_t i = 0; i < key_num; ++i) {
    if (!(((host_keys[i] & 0x7f) > pattern) && (host_scores[i] > threshold))) {
      remaining_keys.push_back(host_keys[i]);
    }
  }
  ASSERT_TRUE(!remaining_keys.empty());

  // 检查这些元素是否仍然存在于表中
  // 创建临时DeviceData用于存储剩余的keys和find结果
  DeviceData<K, V, S> temp_device_data;
  temp_device_data.malloc(remaining_keys.size());

  // 分配普通bool指针内存
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            remaining_keys.size() * sizeof(bool)),
            ACL_ERROR_NONE);

  // 复制剩余的key到设备
  temp_device_data.copy_keys(remaining_keys, remaining_keys.size());

  // 调用find接口检查元素是否存在，第三个参数传递正确的values_ptr指针（即使不使用）
  table->find(remaining_keys.size(), temp_device_data.device_keys,
              temp_device_data.device_values_ptr, temp_device_data.device_found,
              nullptr, temp_device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(temp_device_data.stream), ACL_ERROR_NONE);

  // 复制结果回主机
  ASSERT_EQ(aclrtMemcpy(host_found, remaining_keys.size() * sizeof(bool),
                        temp_device_data.device_found,
                        remaining_keys.size() * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 验证所有应该保留的元素都存在
  for (size_t i = 0; i < remaining_keys.size(); ++i) {
    EXPECT_TRUE(host_found[i])
        << "Key " << remaining_keys[i]
        << " should remain in the table but was not found!";
  }

  // 释放内存
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  host_found = nullptr;

  // 10. 释放资源
  table->clear();
}

TEST(TestErase, test_erase_if) {
  test_erase_if_basic<EraseIfVersion::V1, int64_t, float, uint64_t>();
}

TEST(TestErase, test_erase_if_v2) {
  test_erase_if_basic<EraseIfVersion::V2, int64_t, float, uint64_t>();
}
