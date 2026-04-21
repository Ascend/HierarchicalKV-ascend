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
#include <fstream>
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

class UnlockKeysTest : public ::testing::Test {
 public:
  static constexpr size_t DEFAULT_DIM = 8;
  static constexpr size_t DEFAULT_INIT_CAPACITY = 128UL * 1024;
  static constexpr size_t DEFAULT_HBM_FOR_VALUES = 1UL << 30;

  void SetUp() override {
    init_env();
    
    size_t total_mem = 0;
    size_t free_mem = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem), ACL_ERROR_NONE);
    ASSERT_GT(free_mem, DEFAULT_HBM_FOR_VALUES);
  }

  // 辅助函数：创建和初始化哈希表
  template<typename TableType>
  void InitTable(TableType& table, size_t dim, size_t init_capacity,
                 size_t hbm_for_values) {
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    table.init(options);
  }

  // 辅助函数：验证查询结果
  void VerifyFindResults(const vector<bool>& host_found,
                         const vector<void*>& real_values_ptr,
                         const vector<V>& expected_values,
                         size_t dim,
                         size_t expected_found_num) {
    size_t found_num = 0;
    for (size_t i = 0; i < host_found.size(); i++) {
      if (host_found[i]) {
        ASSERT_NE(real_values_ptr[i], nullptr);
        found_num++;

        // 验证值内容
        vector<V> real_values(dim, 0);
        ASSERT_EQ(aclrtMemcpy(real_values.data(), dim * sizeof(V),
                              real_values_ptr[i], dim * sizeof(V),
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  ACL_ERROR_NONE);
        
        vector<V> expect_values(expected_values.begin() + i * dim,
                                expected_values.begin() + i * dim + dim);
        EXPECT_EQ(expect_values, real_values);
      } else {
        EXPECT_EQ(real_values_ptr[i], nullptr);
      }
    }
    EXPECT_EQ(found_num, expected_found_num);
  }
};

// 用例1: 边界测试 - 空查询 (n=0)
TEST_F(UnlockKeysTest, Unlock0key) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 0;  // 边界值

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  K** locked_key_ptrs = nullptr;
  K* keys = nullptr;
  bool* success = nullptr;

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 调用find，n=0应该直接返回，不执行任何操作
  table.unlock_keys(key_num, locked_key_ptrs, keys, success);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  
  // 验证：函数正常返回，无崩溃
  SUCCEED();
}

// 用例1: 边界测试 - 非空n=1
TEST_F(UnlockKeysTest, Unlock1key) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1;  // 边界值

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  K** locked_key_ptrs = nullptr;
  K* keys = nullptr;
  bool* success = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&locked_key_ptrs), key_num * sizeof(K*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&keys), key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&success), key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 调用find，n=0应该直接返回，不执行任何操作
  table.unlock_keys(key_num, locked_key_ptrs, keys, success);

  bool* host_success = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_success), key_num * sizeof(bool)), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_success, key_num * sizeof(bool), success, key_num * sizeof(bool),
            ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  ASSERT_EQ(*host_success, false);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  
  // 验证：函数正常返回，无崩溃
  SUCCEED();
}

// 用例1: 边界测试 - 非空n=2
TEST_F(UnlockKeysTest, Unlock2key) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 2;  // 边界值

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  K** locked_key_ptrs = nullptr;
  K* keys = nullptr;
  bool* success = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&locked_key_ptrs), key_num * sizeof(K*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  K* locked_keys = *locked_key_ptrs;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&locked_keys), key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&keys), key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&success), key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 调用find，n=0应该直接返回，不执行任何操作
  table.unlock_keys(key_num, locked_key_ptrs, keys, success);

  bool* host_success = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_success), key_num * sizeof(bool)), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_success, key_num * sizeof(bool), success, key_num * sizeof(bool),
            ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  ASSERT_EQ(*host_success, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  
  // 验证：函数正常返回，无崩溃
  SUCCEED();
}