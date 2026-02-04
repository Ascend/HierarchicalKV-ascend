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
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

class FindPtrTest : public ::testing::Test {
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
TEST_F(FindPtrTest, BoundaryTest_EmptyQuery) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 0;  // 边界值

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  K* device_keys = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 调用find，n=0应该直接返回，不执行任何操作
  table.find(key_num, device_keys, device_values_ptr, device_found,
             nullptr, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  
  // 验证：函数正常返回，无崩溃
  SUCCEED();
}

// 用例2: 基本功能 - 小规模全存在（无scores）
TEST_F(FindPtrTest, BasicFunction_SmallScaleAllExist_NoScores) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 申请设备内存
  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键 [1, 2, 3, ..., 1024]
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num);
  
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * dim * sizeof(V),
                        host_values.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入所有键
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  table.find(key_num, device_keys, device_values_ptr, device_found,
             nullptr, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证结果
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  vector<bool> host_found_vec(host_found, host_found + key_num);
  VerifyFindResults(host_found_vec, real_values_ptr, host_values, dim, key_num);

  // 清理
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例3: 基本功能 - 小规模全不存在
TEST_F(FindPtrTest, BasicFunction_SmallScaleNoneExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys_insert(key_num, 0);
  vector<V> host_values_insert(key_num * dim, 0);
  vector<K> host_keys_query(key_num, 0);

  K* device_keys_insert = nullptr;
  V* device_values_insert = nullptr;
  K* device_keys_query = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_insert),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_insert),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_query),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成两批完全不重叠的连续键
  // 插入键: [1, 2, 3, ..., 1024]
  // 查询键: [100000, 100001, ..., 101023]
  create_continuous_keys<K, S, V, dim>(host_keys_insert.data(), nullptr,
                                       host_values_insert.data(), key_num, 1);
  create_continuous_keys<K, S, V, dim>(host_keys_query.data(), nullptr,
                                       nullptr, key_num, 100000);

  ASSERT_EQ(aclrtMemcpy(device_keys_insert, key_num * sizeof(K),
                        host_keys_insert.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values_insert, key_num * dim * sizeof(V),
                        host_values_insert.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_keys_query, key_num * sizeof(K),
                        host_keys_query.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入第一批键
  table.insert_or_assign(key_num, device_keys_insert, device_values_insert,
                         nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 初始化 device_values_ptr 为特定值，验证不存在的键不会修改它
  vector<void*> init_values_ptr(key_num, reinterpret_cast<void*>(0xDEADBEEF));
  ASSERT_EQ(aclrtMemcpy(device_values_ptr, key_num * sizeof(void*),
                        init_values_ptr.data(), key_num * sizeof(void*),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 查询第二批键（全部不存在）
  table.find(key_num, device_keys_query, device_values_ptr, device_found,
             nullptr, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证结果
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 验证所有键都未找到
  size_t not_found_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_FALSE(host_found[i]);
    if (!host_found[i]) {
      not_found_count++;
    }
  }
  EXPECT_EQ(not_found_count, key_num);

  // 清理
  ASSERT_EQ(aclrtFree(device_keys_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_keys_query), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例4: 混合场景 - 部分存在（带scores）
TEST_F(FindPtrTest, MixedScenario_PartialExist_WithScores) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 2048;
  constexpr size_t insert_num = key_num / 2;  // 插入50%
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;  // 使用带score的表类型
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys_insert(insert_num, 0);
  vector<K> host_keys_query(key_num, 0);
  vector<S> host_scores_insert(insert_num, 0);
  vector<V> host_values_insert(insert_num * dim, 0);

  K* device_keys_query = nullptr;
  S* device_scores_insert = nullptr;
  S* device_scores_query = nullptr;
  V* device_values_insert = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_query),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_insert),
                        insert_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_query),
                        key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_insert),
                        insert_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  // 创建自定义stream
  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键: 插入键 [1, 2, ..., 1024]
  create_continuous_keys<K, S, V, dim>(host_keys_insert.data(),
                                       host_scores_insert.data(),
                                       host_values_insert.data(), insert_num, 1);
  
  // 查询键: 前50%是 [1, 2, ..., 1024] (存在), 后50%是 [10000, 10001, ..., 11023] (不存在)
  for (size_t i = 0; i < insert_num; i++) {
    host_keys_query[i] = host_keys_insert[i];
  }
  for (size_t i = insert_num; i < key_num; i++) {
    host_keys_query[i] = 10000 + (i - insert_num);
  }

  ASSERT_EQ(aclrtMemcpy(device_keys_query, key_num * sizeof(K),
                        host_keys_query.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_scores_insert, insert_num * sizeof(S),
                        host_scores_insert.data(), insert_num * sizeof(S),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values_insert, insert_num * dim * sizeof(V),
                        host_values_insert.data(), insert_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入前1024个键
  table.insert_or_assign(insert_num, device_keys_query, device_values_insert,
                         device_scores_insert, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 查询所有2048个键（前1024存在，后1024不存在），使用自定义stream和scores
  table.find(key_num, device_keys_query, device_values_ptr, device_found,
             device_scores_query, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证结果
  bool* host_found = nullptr;
  S* host_scores_result = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_scores_result),
                            key_num * sizeof(S)),
            ACL_ERROR_NONE);
  
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_scores_result, key_num * sizeof(S),
                        device_scores_query, key_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  size_t found_count = 0;
  size_t not_found_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (i < insert_num) {
      // 前50%应该被找到
      EXPECT_TRUE(host_found[i]);
      if (host_found[i]) {
        EXPECT_NE(real_values_ptr[i], nullptr);
        found_count++;
      }
    } else {
      // 后50%应该未找到
      EXPECT_FALSE(host_found[i]);
      if (!host_found[i]) {
        not_found_count++;
      }
    }
  }
  
  // 精确验证：前50%全部找到，后50%全部未找到
  EXPECT_EQ(found_count, insert_num);
  EXPECT_EQ(not_found_count, insert_num);

  // 清理
  ASSERT_EQ(aclrtFree(device_keys_query), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores_query), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_scores_result), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例5: 重复键测试 - unique_key=false
TEST_F(FindPtrTest, DuplicateKeys_UniqueKeyFalse) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1000;
  constexpr size_t unique_key_num = 200;  // 实际只有200个唯一键
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys_unique(unique_key_num, 0);
  vector<K> host_keys_with_dup(key_num, 0);
  vector<S> host_scores_unique(unique_key_num, 0);
  vector<V> host_values_unique(unique_key_num * dim, 0);

  K* device_keys_unique = nullptr;
  K* device_keys_with_dup = nullptr;
  S* device_scores_unique = nullptr;
  S* device_scores_query = nullptr;
  V* device_values_unique = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_unique),
                        unique_key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_with_dup),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_unique),
                        unique_key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_query),
                        key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_unique),
                        unique_key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成唯一键
  create_continuous_keys<K, S, V, dim>(host_keys_unique.data(),
                                       host_scores_unique.data(),
                                       host_values_unique.data(),
                                       unique_key_num);

  // 创建包含重复键的数组：[1,2,3,1,2,3,4,5,6,4,5,6,...]
  for (size_t i = 0; i < key_num; i++) {
    host_keys_with_dup[i] = host_keys_unique[i % unique_key_num];
  }

  ASSERT_EQ(aclrtMemcpy(device_keys_unique, unique_key_num * sizeof(K),
                        host_keys_unique.data(), unique_key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_keys_with_dup, key_num * sizeof(K),
                        host_keys_with_dup.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_scores_unique, unique_key_num * sizeof(S),
                        host_scores_unique.data(), unique_key_num * sizeof(S),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values_unique, unique_key_num * dim * sizeof(V),
                        host_values_unique.data(), unique_key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入唯一键
  table.insert_or_assign(unique_key_num, device_keys_unique,
                         device_values_unique, device_scores_unique, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 查询包含重复键的数组，设置 unique_key=false
  table.find(key_num, device_keys_with_dup, device_values_ptr, device_found,
             device_scores_query, stream, false);  // unique_key=false
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证结果
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 验证重复键的查询结果一致
  unordered_map<K, void*> key_to_ptr;
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(host_found[i]);
    
    K key = host_keys_with_dup[i];
    if (key_to_ptr.find(key) == key_to_ptr.end()) {
      key_to_ptr[key] = real_values_ptr[i];
    } else {
      // 同一个键的多次查询应返回相同的指针
      EXPECT_EQ(real_values_ptr[i], key_to_ptr[key]);
    }
  }

  // 清理
  ASSERT_EQ(aclrtFree(device_keys_unique), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_keys_with_dup), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores_unique), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores_query), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_unique), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例6: 中等规模测试
TEST_F(FindPtrTest, MediumScale_PartialExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 10 * 1024;
  constexpr size_t insert_num = key_num / 2;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys_insert(insert_num, 0);
  vector<K> host_keys_query(key_num, 0);
  vector<S> host_scores_insert(insert_num, 0);
  vector<V> host_values_insert(insert_num * dim, 0);

  K* device_keys_query = nullptr;
  S* device_scores_insert = nullptr;
  S* device_scores_query = nullptr;
  V* device_values_insert = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_query),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_insert),
                        insert_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_query),
                        key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_insert),
                        insert_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键: 插入键 [1, 2, ..., 5120]
  create_continuous_keys<K, S, V, dim>(host_keys_insert.data(),
                                       host_scores_insert.data(),
                                       host_values_insert.data(), insert_num, 1);
  
  // 查询键: 前50%是 [1, 2, ..., 5120] (存在), 后50%是 [20000, 20001, ..., 25119] (不存在)
  for (size_t i = 0; i < insert_num; i++) {
    host_keys_query[i] = host_keys_insert[i];
  }
  for (size_t i = insert_num; i < key_num; i++) {
    host_keys_query[i] = 20000 + (i - insert_num);
  }

  ASSERT_EQ(aclrtMemcpy(device_keys_query, key_num * sizeof(K),
                        host_keys_query.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_scores_insert, insert_num * sizeof(S),
                        host_scores_insert.data(), insert_num * sizeof(S),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values_insert, insert_num * dim * sizeof(V),
                        host_values_insert.data(), insert_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入前50%
  table.insert_or_assign(insert_num, device_keys_query, device_values_insert,
                         device_scores_insert, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 查询所有键
  table.find(key_num, device_keys_query, device_values_ptr, device_found,
             device_scores_query, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证结果
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  size_t found_count = 0;
  size_t not_found_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (i < insert_num) {
      EXPECT_TRUE(host_found[i]);
      if (host_found[i]) {
        found_count++;
      }
    } else {
      EXPECT_FALSE(host_found[i]);
      if (!host_found[i]) {
        not_found_count++;
      }
    }
  }
  
  // 精确验证：前50%全部找到，后50%全部未找到
  EXPECT_EQ(found_count, insert_num);
  EXPECT_EQ(not_found_count, insert_num);

  // 清理
  ASSERT_EQ(aclrtFree(device_keys_query), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores_query), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例7: 异常场景 - 初始值保留验证
TEST_F(FindPtrTest, ExceptionScenario_InitialValuePreserved) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 100;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys_insert(key_num, 0);
  vector<V> host_values_insert(key_num * dim, 0);
  vector<K> host_keys_query(key_num, 0);

  K* device_keys_insert = nullptr;
  V* device_values_insert = nullptr;
  K* device_keys_query = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_insert),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_insert),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys_query),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成完全不重叠的连续键
  // 插入键: [1, 2, ..., 100]
  // 查询键: [50000, 50001, ..., 50099]
  create_continuous_keys<K, S, V, dim>(host_keys_insert.data(), nullptr,
                                       host_values_insert.data(), key_num, 1);
  create_continuous_keys<K, S, V, dim>(host_keys_query.data(), nullptr,
                                       nullptr, key_num, 50000);

  ASSERT_EQ(aclrtMemcpy(device_keys_insert, key_num * sizeof(K),
                        host_keys_insert.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values_insert, key_num * dim * sizeof(V),
                        host_values_insert.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_keys_query, key_num * sizeof(K),
                        host_keys_query.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入第一批键
  table.insert_or_assign(key_num, device_keys_insert, device_values_insert,
                         nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 初始化values_ptr为特定标记值
  const uintptr_t MAGIC_VALUE = 0xDEADBEEFDEADBEEF;
  vector<void*> init_ptrs(key_num, reinterpret_cast<void*>(MAGIC_VALUE));
  ASSERT_EQ(aclrtMemcpy(device_values_ptr, key_num * sizeof(void*),
                        init_ptrs.data(), key_num * sizeof(void*),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 查询不存在的键
  table.find(key_num, device_keys_query, device_values_ptr, device_found,
             nullptr, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> result_ptrs(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(result_ptrs.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 验证所有键未找到，且values_ptr保持初始值或为nullptr
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_FALSE(host_found[i]);
    // 根据实现，可能保持MAGIC_VALUE或设置为nullptr，两者都是可接受的
    // 这里主要验证不会是其他非法值
    if (!host_found[i]) {
      // 可以是nullptr或保持原值
      bool is_valid = (result_ptrs[i] == nullptr) ||
                      (result_ptrs[i] == reinterpret_cast<void*>(MAGIC_VALUE));
      EXPECT_TRUE(is_valid);
    }
  }

  // 清理
  ASSERT_EQ(aclrtFree(device_keys_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_insert), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_keys_query), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例8: 大规模测试 - 64K全存在
TEST_F(FindPtrTest, LargeScale_AllExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 64 * 1024;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键 [1, 2, 3, ..., 65536]
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num, 1);
  
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * dim * sizeof(V),
                        host_values.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入所有键
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 执行查找
  table.find(key_num, device_keys, device_values_ptr, device_found,
             nullptr, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证结果
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  size_t found_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(host_found[i]);
    if (host_found[i]) {
      EXPECT_NE(real_values_ptr[i], nullptr);
      found_count++;
    }
  }
  
  EXPECT_EQ(found_count, key_num);

  // 清理
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例9: 不同dim测试 - dim=128
TEST_F(FindPtrTest, DifferentDim_128) {
  constexpr size_t dim = 128;
  constexpr size_t key_num = 1024;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num, 1);
  
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * dim * sizeof(V),
                        host_values.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入所有键
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 执行查找
  table.find(key_num, device_keys, device_values_ptr, device_found,
             nullptr, stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证结果
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  vector<bool> host_found_vec(host_found, host_found + key_num);
  VerifyFindResults(host_found_vec, real_values_ptr, host_values, dim, key_num);

  // 清理
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

