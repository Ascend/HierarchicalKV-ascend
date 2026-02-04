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
#include <cstddef>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

TEST(test_find_or_insert, basic_function) {
  // 1. 初始化
  init_env();

  size_t free_mem = 0;
  size_t total_mem = 0;
  constexpr size_t hbm_for_values = 1UL << 30;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem << "need:" << hbm_for_values;

  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t key_num = 1UL * 1024;

  using K = uint64_t;
  using V = float;
  using S = uint64_t;
  size_t each_key_size = sizeof(K);
  size_t each_value_size = sizeof(V);
  size_t each_score_size = sizeof(S);

  // 2. 建表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  using Table = HashTable<K, V>;

  Table table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  // 3. 数据准备
  // 3.1 host数据
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 3.2 申请hbm内存
  K* device_keys = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * each_key_size, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 4. 插值
  // 4.1 生产连续值
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr,
                                       key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * each_key_size, host_keys.data(),
                        key_num * each_key_size, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 4.2 下发算子
  table.find_or_insert(key_num, device_keys, device_values_ptr, device_found,
                       nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);

  // 4.3 写值
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  size_t found_num = 0;
  size_t refused_num = 0;
  size_t insert_num = 0;
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  vector<void*> expect_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(expect_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      ASSERT_NE(expect_values_ptr[i], nullptr);
      vector<V> expect_values(dim, i);
      ASSERT_EQ(aclrtMemcpy(expect_values_ptr[i], dim * each_value_size,
                            expect_values.data(), dim * each_value_size,
                            ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
      found_num++;
    } else {
      if (expect_values_ptr[i] == nullptr) {
        refused_num++;
      } else {
        vector<V> expect_values(dim, i);
        ASSERT_EQ(aclrtMemcpy(expect_values_ptr[i], dim * each_value_size,
                              expect_values.data(), dim * each_value_size,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  ACL_ERROR_NONE);
        insert_num++;
      }
    }
  }
  EXPECT_EQ(found_num, 0);
  EXPECT_EQ(refused_num, 0);
  EXPECT_EQ(insert_num, key_num);

  // 5. 校验结果
  ASSERT_EQ(
      aclrtMemset(device_values_ptr, key_num * sizeof(V*), 0, key_num * sizeof(V*)),
      ACL_ERROR_NONE);
  table.find_or_insert(key_num, device_keys, device_values_ptr, device_found,
                       nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);

  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(
      aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*), device_values_ptr,
                  key_num * sizeof(void*), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);
  // 5.1 values指针相同
  EXPECT_EQ(real_values_ptr, expect_values_ptr);

  bool* host_found_again = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found_again),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found_again, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  // 5.2 结果相同
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_NE(host_found[i], host_found_again[i]);
    if (host_found_again[i]) {
      ASSERT_NE(real_values_ptr[i], nullptr);
      vector<V> expect_values(dim, i);
      vector<V> real_values(dim, 0);
      ASSERT_EQ(
          aclrtMemcpy(real_values.data(), dim * each_value_size, real_values_ptr[i],
                      dim * each_value_size, ACL_MEMCPY_HOST_TO_DEVICE),
          ACL_ERROR_NONE);
      EXPECT_EQ(real_values, expect_values);
    } else {
      EXPECT_EQ(real_values_ptr[i], nullptr);
    }
  }

  // 6. 释放内存
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found_again), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
}

TEST(test_find_or_insert, empty_table_for_sim) {
  // 1. 初始化
  init_env();

  size_t free_mem = 0;
  size_t total_mem = 0;
  constexpr size_t hbm_for_values = 1UL << 30;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem << "need:" << hbm_for_values;

  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 1024;
  constexpr size_t key_num = 100;

  using K = uint64_t;
  using V = float;
  using S = uint64_t;
  size_t each_key_size = sizeof(K);
  size_t each_value_size = sizeof(V);
  size_t each_score_size = sizeof(S);

  // 2. 建表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
      .num_of_buckets_per_alloc = init_capacity / 128,
  };
  using Table = HashTable<K, V>;

  Table table;
  table.init(options);

  // 3. 数据准备
  // 3.1 host数据
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 3.2 申请hbm内存
  K* device_keys = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * each_key_size, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 4. 插值
  // 4.1 生产连续值
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr,
                                       key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * each_key_size, host_keys.data(),
                        key_num * each_key_size, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 4.2 下发算子
  table.find_or_insert(key_num, device_keys, device_values_ptr, device_found,
                       nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  table.find_or_insert(key_num, device_keys, device_values_ptr, device_found,
                       nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 4.3 写值
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  size_t found_num = 0;
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  vector<void*> expect_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(expect_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      ASSERT_NE(expect_values_ptr[i], nullptr);
      vector<V> expect_values(dim, i);
      ASSERT_EQ(aclrtMemcpy(expect_values_ptr[i], dim * each_value_size,
                            expect_values.data(), dim * each_value_size,
                            ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
      found_num++;
    } else {
      EXPECT_EQ(expect_values_ptr[i], nullptr);
    }
  }
  ASSERT_NE(found_num, 0);

  // 6. 释放内存
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
}