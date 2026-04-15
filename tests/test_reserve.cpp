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
#include <limits>
#include <memory>
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

void test_reserve_basic(size_t dim, size_t key_num, size_t init_capacity,
                        size_t max_capacity, size_t max_hbm_for_vectors) {
  // 1. 初始化
  init_env();

  size_t total_mem = 0;
  size_t free_mem = 0;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, max_hbm_for_vectors)
      << "free HBM is not enough free:" << free_mem
      << "need:" << max_hbm_for_vectors;

  // 2. 建表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = max_hbm_for_vectors,
      .dim = dim,
      .max_load_factor = 1.0f,
  };
  using Table = HashTable<K, V>;

  Table table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 3. 使用DeviceData管理设备内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num, dim);

  // 4. 插值
  // 4.1 生产连续值
  create_continuous_keys<K, S, V>(dim, host_keys.data(), nullptr,
                                  host_values.data(), key_num);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num, dim);

  // 4.2 插值
  table.insert_or_assign(key_num, device_data.device_keys,
                         device_data.device_values, nullptr,
                         device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(device_data.stream), key_num);

  // 5. 验证reserve
  auto cur_capacity = table.capacity();
  auto load_factor = table.load_factor(device_data.stream);
  EXPECT_EQ(cur_capacity, init_capacity);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 1.0));
  EXPECT_EQ(table.size(device_data.stream), key_num);
  // 5.1 不能缩小
  table.reserve(cur_capacity / 2, device_data.stream);
  load_factor = table.load_factor(device_data.stream);
  cur_capacity = table.capacity();
  EXPECT_EQ(cur_capacity, init_capacity);
  EXPECT_EQ(table.size(device_data.stream), key_num);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 1.0));
  // 5.2 容量满足不扩容
  table.reserve(cur_capacity);
  load_factor = table.load_factor(device_data.stream);
  cur_capacity = table.capacity();
  EXPECT_EQ(cur_capacity, init_capacity);
  EXPECT_EQ(table.size(device_data.stream), key_num);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 1.0));
  // 5.3 按2倍扩容
  table.reserve(cur_capacity + 1, device_data.stream);
  cur_capacity = table.capacity();
  load_factor = table.load_factor(device_data.stream);
  EXPECT_EQ(cur_capacity, init_capacity * 2);
  EXPECT_EQ(load_factor, (key_num * 1.0) / (init_capacity * 2.0));
  EXPECT_EQ(table.size(device_data.stream), key_num);
  // 5.4 按最大容量扩容
  size_t target_capacity = init_capacity * 2;
  while (target_capacity <= max_capacity) {
    table.reserve(target_capacity);
    cur_capacity = table.capacity();
    load_factor = table.load_factor(device_data.stream);
    EXPECT_EQ(cur_capacity, target_capacity);
    EXPECT_EQ(load_factor, (key_num * 1.0) / (target_capacity * 1.0));
    EXPECT_EQ(table.size(device_data.stream), key_num);

    // 6. 校验扩容后，value不变
    table.find(key_num, device_data.device_keys, device_data.device_values_ptr,
               device_data.device_found, nullptr, device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

    bool* host_found = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                              key_num * sizeof(bool)),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool),
                          device_data.device_found, key_num * sizeof(bool),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    vector<void*> real_values_ptr(key_num, nullptr);
    ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                          device_data.device_values_ptr,
                          key_num * sizeof(void*), ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    size_t found_num = 0;
    for (size_t i = 0; i < key_num; i++) {
      if (host_found[i]) {
        ASSERT_NE(real_values_ptr[i], nullptr);
        found_num++;

        vector<V> real_values(dim, 0);
        ASSERT_EQ(
            aclrtMemcpy(real_values.data(), dim * device_data.each_value_size,
                        real_values_ptr[i], dim * device_data.each_value_size,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
        vector<V> expect_values(host_values.begin() + i * dim,
                                host_values.begin() + i * dim + dim);
        uint64_t hash = test_util::Murmur3HashHost(host_keys[i]);
        hash = hash % (table.capacity() / 2);
        uint64_t bkt_idx = hash / table.max_bucket_size();
        uint64_t key_pos = hash % table.max_bucket_size();
        ASSERT_EQ(expect_values, real_values)
            << "key id :" << host_keys[i] << " bkt_idx:" << bkt_idx
            << " key_pos:" << key_pos;
      } else {
        EXPECT_EQ(real_values_ptr[i], nullptr);
      }
    }
    ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
    EXPECT_EQ(found_num, key_num);
    target_capacity *= 2;
  }
  table.clear(device_data.stream);
  EXPECT_EQ(table.size(device_data.stream), 0);
}

TEST(test_reserve, test_reserve_all_hbm) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1UL * 1024;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 128;
  constexpr size_t hbm_for_values = 1UL << 30;
  test_reserve_basic(dim, key_num, init_capacity, max_capacity, hbm_for_values);
}

TEST(test_reserve, test_reserve_to_ddr) {
  constexpr size_t dim = 16;
  constexpr size_t key_num = 2UL * 1024;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 128;
  size_t max_hbm_for_vectors = init_capacity * 2 * dim * sizeof(V);
  test_reserve_basic(dim, key_num, init_capacity, max_capacity,
                     max_hbm_for_vectors);
}

void test_reserve_with_defragmentation(size_t dim, size_t key_num,
                                       size_t init_capacity,
                                       size_t max_capacity,
                                       size_t max_hbm_for_vectors) {
  // 1. 初始化
  init_env();

  size_t total_mem = 0;
  size_t free_mem = 0;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, max_hbm_for_vectors)
      << "free HBM is not enough free:" << free_mem
      << "need:" << max_hbm_for_vectors;

  // 2. 建表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = max_capacity,
      .max_hbm_for_vectors = max_hbm_for_vectors,
      .dim = dim,
      .max_load_factor = numeric_limits<float>::max(),
  };
  using Table = HashTable<K, V>;

  Table table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 3. 使用DeviceData管理设备内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num, dim);

  // 4. 插值
  // 4.1 生产连续值
  create_continuous_keys<K, S, V>(dim, host_keys.data(), nullptr,
                                  host_values.data(), key_num);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num, dim);

  // 4.2 插值
  uint32_t add_time = key_num / init_capacity;
  for (uint32_t i = 0; i < add_time; i++) {
    table.insert_or_assign(init_capacity,
                           device_data.device_keys + i * init_capacity,
                           device_data.device_values + i * init_capacity * dim,
                           nullptr, device_data.stream);
  }
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(device_data.stream), init_capacity);

  // 5. 验证reserve
  auto cur_capacity = table.capacity();
  auto load_factor = table.load_factor(device_data.stream);
  EXPECT_EQ(cur_capacity, init_capacity);
  EXPECT_EQ(load_factor, 1.0f);
  // 5.1 按2倍扩容
  table.reserve(cur_capacity + 1, device_data.stream);
  cur_capacity = table.capacity();
  load_factor = table.load_factor(device_data.stream);
  EXPECT_EQ(cur_capacity, init_capacity * 2);
  EXPECT_EQ(load_factor, 0.5f);
  EXPECT_EQ(table.size(device_data.stream), init_capacity);

  // 6. 校验扩容后，value不变
  table.find(key_num, device_data.device_keys, device_data.device_values_ptr,
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
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_data.device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      ASSERT_NE(real_values_ptr[i], nullptr);
      found_num++;

      vector<V> real_values(dim, 0);
      ASSERT_EQ(
          aclrtMemcpy(real_values.data(), dim * device_data.each_value_size,
                      real_values_ptr[i], dim * device_data.each_value_size,
                      ACL_MEMCPY_DEVICE_TO_HOST),
          ACL_ERROR_NONE);
      vector<V> expect_values(host_values.begin() + i * dim,
                              host_values.begin() + i * dim + dim);
      ASSERT_EQ(expect_values, real_values);
    } else {
      EXPECT_EQ(real_values_ptr[i], nullptr);
    }
  }
  EXPECT_EQ(found_num, init_capacity);

  table.clear(device_data.stream);
  EXPECT_EQ(table.size(device_data.stream), 0);

  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
}

TEST(test_reserve, test_reserve_all_hbm_with_defragmentation) {
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 128;
  constexpr size_t key_num = max_capacity * 2;
  constexpr size_t hbm_for_values = 1UL << 30;
  test_reserve_with_defragmentation(dim, key_num, init_capacity, max_capacity,
                                    hbm_for_values);
}

TEST(test_reserve, test_reserve_to_ddr_with_defragmentation) {
  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t max_capacity = init_capacity * 128;
  constexpr size_t key_num = max_capacity * 2;
  size_t max_hbm_for_vectors = init_capacity * dim * sizeof(V);
  test_reserve_with_defragmentation(dim, key_num, init_capacity, max_capacity,
                                    max_hbm_for_vectors);
}