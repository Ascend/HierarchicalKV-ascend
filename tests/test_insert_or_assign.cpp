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
#include <memory>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

constexpr size_t DEFAULT_DIM = 8;
constexpr size_t DEFAULT_CAPACITY = 128UL * 1024;

template <typename K, typename V, typename S>
unique_ptr<HashTable<K, V, S>> get_table(size_t dim, size_t capacity,
                                         size_t num_of_buckets_per_alloc) {
  using Table = HashTable<K, V, S>;
  auto table = make_unique<Table>();

  size_t total_mem = 0;
  size_t free_mem = 0;
  constexpr size_t hbm_for_values = 4UL << 30;
  EXPECT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  EXPECT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem
      << "need:" << hbm_for_values;

  HashTableOptions options{
      .init_capacity = capacity,
      .max_capacity = capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
      .num_of_buckets_per_alloc = num_of_buckets_per_alloc,
  };

  table->init(options);

  return table;
}

template <typename K, typename V, typename S>
unique_ptr<HashTable<K, V, S>> get_default_table() {
  return get_table<K, V, S>(DEFAULT_DIM, DEFAULT_CAPACITY, 1);
}

template <typename K, typename V, typename S>
class DeviceData {
 public:
  DeviceData() = default;

  ~DeviceData() {
    if (device_keys != nullptr) {
      EXPECT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
  }

  void malloc(size_t key_num, size_t dim = DEFAULT_DIM) {
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          key_num * each_key_size, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&device_values),
                    key_num * each_value_size * dim, ACL_MEM_MALLOC_HUGE_FIRST),
        ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                          key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                          key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  }

  void copy_keys(vector<K>& host_keys, size_t key_num) {
    ASSERT_EQ(
        aclrtMemcpy(device_keys, key_num * each_key_size, host_keys.data(),
                    key_num * each_key_size, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_ERROR_NONE);
  }

  void copy_values(vector<V>& host_values, size_t key_num,
                   size_t dim = DEFAULT_DIM) {
    ASSERT_EQ(aclrtMemcpy(device_values, key_num * each_value_size * dim,
                          host_values.data(), key_num * each_value_size * dim,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  aclrtStream stream = nullptr;
  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  size_t each_key_size = sizeof(K);
  size_t each_score_size = sizeof(S);
  size_t each_value_size = sizeof(V);
};

template <typename K, typename V, typename S>
void check_result(vector<V>& host_values, size_t key_num,
                  DeviceData<K, V, S>& device_data,
                  size_t expect_found_num = numeric_limits<size_t>::max(), size_t dim = DEFAULT_DIM) {
  expect_found_num = expect_found_num == numeric_limits<size_t>::max()
                         ? key_num
                         : expect_found_num;
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMemcpy(host_found, key_num * sizeof(bool), device_data.device_found,
                  key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_data.device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  size_t found_num = 0;
  vector<V> real_values(dim, 0);
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      ASSERT_NE(real_values_ptr[i], nullptr);
      found_num++;

      ASSERT_EQ(aclrtMemcpy(real_values.data(),
                            dim * device_data.each_value_size,
                            real_values_ptr[i],
                            dim * device_data.each_value_size,
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      vector<V> expect_values(
          host_values.begin() + i * dim,
          host_values.begin() + i * dim + dim);
      ASSERT_EQ(expect_values, real_values);
    } else {
      ASSERT_EQ(real_values_ptr[i], nullptr);
    }
  }
  EXPECT_EQ(found_num, expect_found_num);

  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
}

template <typename K, typename V, typename S>
void test_insert_and_assign_basic() {
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
  vector<V> host_values(key_num * DEFAULT_DIM, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               host_values.data(), key_num);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 5. 校验插入结果
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  check_result(host_values, key_num, device_data);

  // 6. 插入重复key，更新value值，验证更新功能
  create_continuous_keys<K, S, V, DEFAULT_DIM>(
      host_keys.data(), nullptr, host_values.data(), key_num, 1 + key_num);
  device_data.copy_values(host_values, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  // 7. 再次校验
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  check_result(host_values, key_num, device_data);

  table->clear(device_data.stream);
  EXPECT_EQ(table->size(device_data.stream), 0);
}

TEST(test_insert_or_assign, test_insert_and_assign) {
  test_insert_and_assign_basic<uint64_t, float, uint64_t>();
  test_insert_and_assign_basic<uint64_t, double, uint64_t>();
  test_insert_and_assign_basic<int64_t, uint16_t, uint64_t>();
  test_insert_and_assign_basic<int64_t, uint8_t, uint64_t>();
}

TEST(test_insert_or_assign, test_n_greater_than_thread_all) {
  // 1. 初始化
  init_env();

  using K = int64_t;
  using V = float;
  using S = uint64_t;

  auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  HKV_CHECK(platform != nullptr, "get platform failed.");
  auto block_dim = platform->GetCoreNumAiv();
  // 这里512是和insert_or_assign算子中使用的thread_num保持一致
  uint64_t thread_all = 512 * block_dim;
  // 保证算子内部可以循环两次，验证第二次循环功能正常
  uint64_t key_num = 2 * thread_all;

  // 2. 申请device内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 3. 创建数据
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * DEFAULT_DIM, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(
      host_keys.data(), nullptr, host_values.data(), key_num, key_num - 100);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);

  // 4. 建表
  auto table = get_default_table<K, V, S>();
  EXPECT_EQ(table->size(), 0);

  // 5. 插入大量key，保证算子内部线程触发多次循环
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 6. 校验值插入正常
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  check_result(host_values, key_num, device_data);
  table->clear();

  uint64_t valid_num = 100;
  // 7. 设置第一次循环key为无效值
  create_continuous_keys<K, S, V, DEFAULT_DIM>(
      host_keys.data(), nullptr, host_values.data(), key_num, key_num * 2);
  for (size_t i = 0; i < key_num - valid_num; i++) {
    host_keys[i] = DEFAULT_RESERVED_KEY_MASK;
  }
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);

  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), valid_num);

  // 8. 确认有效部分已经写入
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  check_result(host_values, key_num, device_data, valid_num);

  table->clear();
}

TEST(test_insert_or_assign, test_loader_factor_to_1) {
  // 1. 初始化
  init_env();
  using K = int64_t;
  using V = float;
  using S = uint64_t;

  // 2. 建表
  auto table = get_default_table<K, V, S>();
  EXPECT_EQ(table->size(), 0);

  constexpr size_t key_num = 1UL * 1024;
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * DEFAULT_DIM, 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  uint32_t insert_time = 0;
  // 实测正常情况下需要插166次
  constexpr uint32_t expect_insert_time = 166;
  // 4. 循环插值，直至满足负载率要求
  while (table->load_factor() != 1.0) {
    create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                                 host_values.data(), key_num,
                                                 key_num * insert_time);
    device_data.copy_keys(host_keys, key_num);
    device_data.copy_values(host_values, key_num);

    table->insert_or_assign(key_num, device_data.device_keys,
                            device_data.device_values, nullptr,
                            device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
    insert_time++;
    // 避免死循环插入卡死
    ASSERT_LE(insert_time, expect_insert_time);
  }
  EXPECT_EQ(insert_time, expect_insert_time);

  table->clear();
  insert_time = 0;
  while (table->load_factor() != 1.0) {
    create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                                 host_values.data(), key_num,
                                                 key_num * insert_time);
    device_data.copy_keys(host_keys, key_num);
    device_data.copy_values(host_values, key_num);

    table->insert_or_assign(key_num, device_data.device_keys,
                            device_data.device_values, nullptr,
                            device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
    insert_time++;
    // 避免死循环插入卡死
    ASSERT_LE(insert_time, expect_insert_time);
  }
  EXPECT_EQ(insert_time, expect_insert_time);
}

TEST(test_insert_or_assign, test_evict) {
  // 1. 初始化
  init_env();
  using K = int64_t;
  using V = float;
  using S = uint64_t;

  // 2. 建表
  constexpr size_t capacity = 128UL;
  auto table = get_table<K, V, S>(DEFAULT_DIM, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  constexpr size_t key_num = 1024;
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 空桶插值
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * DEFAULT_DIM, 0);
  create_continuous_keys<K, S, V, DEFAULT_DIM>(host_keys.data(), nullptr,
                                               host_values.data(), key_num, 0);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity);

  // 5. 校验插入结果
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  check_result(host_values, key_num, device_data, capacity);
}

TEST(test_insert_or_assign, test_dim) {
  // 1. 初始化
  init_env();
  using K = int64_t;
  using V = float;
  using S = uint64_t;

  // 2. 建表
  constexpr size_t dim = 27;
  constexpr size_t capacity = 128UL;
  auto table = get_table<K, V, S>(dim, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  constexpr size_t key_num = 1024;
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num, dim);

  // 4. 空桶插值
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num, 0);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num, dim);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), capacity);

  // 5. 校验插入结果
  table->find(key_num, device_data.device_keys, device_data.device_values_ptr,
              device_data.device_found, nullptr, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  check_result(host_values, key_num, device_data, capacity, dim);
}

TEST(test_insert_or_assign, test_little_demo_benchmark) {
  // 1. 初始化
  init_env();
  using K = int64_t;
  using V = float;
  using S = uint64_t;

  // 2. 建表
  constexpr size_t dim = 8;
  constexpr size_t capacity = 128UL * 10;
  auto table = get_table<K, V, S>(dim, capacity, 8);

  constexpr size_t key_num = 128 * 5;
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num, dim);

  // 4. 循环插值，直至满足load_factor要求
  float target_load_factor = 0.5f;
  K start = 0;
#ifdef SIM_MODE
  unordered_map<float, vector<int64_t>> add_num_map{
      {0.5, {640}},
      {0.75, {640, 320}},
      {1.0, {640, 640, 90}},
  };
  auto it = add_num_map.find(target_load_factor);
  EXPECT_NE(it, add_num_map.end());
  auto& add_num = it->second;
  for (auto key_num_append : add_num) {
    create_continuous_keys<K, S, V, dim>(
        host_keys.data(), nullptr, host_values.data(), key_num_append, start);
    device_data.copy_keys(host_keys, key_num);
    device_data.copy_values(host_values, key_num, dim);
    table->find_or_insert(
        key_num_append, device_data.device_keys, device_data.device_values_ptr,
        device_data.device_found, nullptr, device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
    start += key_num_append;
  }
#else
  float real_load_factor = table->load_factor(device_data.stream);
  const float epsilon = 0.001f;
  cout << "target_load_factor : " << target_load_factor << endl;
  while (target_load_factor - real_load_factor > epsilon) {
    auto key_num_append = static_cast<int64_t>(
        (target_load_factor - real_load_factor) * capacity);
    if (key_num_append <= 0) {
      break;
    }
    key_num_append = min(static_cast<int64_t>(key_num), key_num_append);
    cout << "key_num_append : " << key_num_append << endl;
    create_continuous_keys<K, S, V, dim>(
        host_keys.data(), nullptr, host_values.data(), key_num_append, start);
    device_data.copy_keys(host_keys, key_num);
    device_data.copy_values(host_values, key_num, dim);
    table->insert_or_assign(key_num_append, device_data.device_keys,
                            device_data.device_values, nullptr,
                            device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
    start += key_num_append;
    real_load_factor = table->load_factor(device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  }
#endif

  const float hit_rate = 0.6f;
  uint32_t test_num = capacity / 128;
  uint32_t divide = static_cast<uint32_t>(test_num * hit_rate);
  for (uint32_t i = divide; i < test_num; i++) {
    host_keys[i] = numeric_limits<K>::max() - i;
  }
  device_data.copy_keys(host_keys, test_num);
  aclrtMemset(device_data.device_values,
              test_num * device_data.each_value_size * dim, 2,
              test_num * device_data.each_value_size * dim);
  table->insert_or_assign(test_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
}
