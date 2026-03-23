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
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "test_find_base.h"

// find(value copy) 专用测试类
class FindValueTest : public FindTestBase {
 public:
  // 验证 find 返回的 values / founds / scores
  template<size_t DIM>
  void VerifyFindResults(const vector<K>& query_keys,
                         const vector<V>& result_values,
                         const vector<bool>& result_founds,
                         const vector<V>& expected_values,
                         const unordered_map<K, size_t>& key_to_insert_idx,
                         size_t expected_found_num,
                         size_t expected_not_found_num,
                         const vector<S>* result_scores = nullptr,
                         const vector<S>* expected_scores = nullptr) {
    size_t found_num = 0;
    size_t not_found_num = 0;
    for (size_t i = 0; i < query_keys.size(); i++) {
      auto it = key_to_insert_idx.find(query_keys[i]);
      if (it != key_to_insert_idx.end()) {
        EXPECT_TRUE(result_founds[i]) << "key=" << query_keys[i] << " should be found";
        if (result_founds[i]) {
          found_num++;
          size_t insert_idx = it->second;
          for (size_t j = 0; j < DIM; j++) {
            EXPECT_EQ(result_values[i * DIM + j],
                      expected_values[insert_idx * DIM + j])
                << "key=" << query_keys[i] << " dim_idx=" << j;
          }
          if (result_scores && expected_scores) {
            EXPECT_EQ((*result_scores)[i], (*expected_scores)[insert_idx])
                << "key=" << query_keys[i] << " score mismatch";
          }
        }
      } else {
        EXPECT_FALSE(result_founds[i]) << "key=" << query_keys[i] << " should NOT be found";
        if (!result_founds[i]) {
          not_found_num++;
        }
      }
    }
    EXPECT_EQ(found_num, expected_found_num);
    EXPECT_EQ(not_found_num, expected_not_found_num);
  }
};

// 用例1: 边界测试 - 空查询 (n=0)
TEST_F(FindValueTest, BoundaryTest_EmptyQuery) {
  constexpr size_t dim = DEFAULT_DIM;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  K* device_keys = nullptr;
  V* device_values = nullptr;
  bool* device_found = nullptr;
  table.find(0, device_keys, device_values, device_found, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  SUCCEED();
}

// 用例2: 基本功能 - 小规模全存在（无scores）
TEST_F(FindValueTest, BasicFunction_SmallScaleAllExist_NoScores) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> host_keys;
  vector<V> host_values;
  vector<S> host_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, key_num, host_keys, host_values, host_scores, false, stream);

  // 构建key→索引映射
  unordered_map<K, size_t> key_to_idx;
  for (size_t i = 0; i < key_num; i++) {
    key_to_idx[host_keys[i]] = i;
  }

  // 分配查询用设备内存
  K* device_keys = nullptr;
  V* device_values = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  table.find(key_num, device_keys, device_values, device_found,
             nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 回收结果
  vector<V> result_values(key_num * dim);
  vector<bool> result_founds(key_num);
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(result_values.data(), key_num * dim * sizeof(V),
                        device_values, key_num * dim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  result_founds.assign(host_found, host_found + key_num);

  VerifyFindResults<dim>(host_keys, result_values, result_founds,
                         host_values, key_to_idx, key_num, 0);

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例3: 基本功能 - 小规模全不存在
TEST_F(FindValueTest, BasicFunction_SmallScaleNoneExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 插入键 [1, 1024]
  vector<K> insert_keys;
  vector<V> insert_values;
  vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, key_num, insert_keys, insert_values, insert_scores, false, stream);

  // 查询键 [100000, 101023]，全部不存在
  vector<K> query_keys(key_num);
  create_continuous_keys<K, S, V, dim>(query_keys.data(), nullptr,
                                       nullptr, key_num, 100000);

  K* device_keys = nullptr;
  V* device_values = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), query_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  table.find(key_num, device_keys, device_values, device_found,
             nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  size_t not_found_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_FALSE(host_found[i]);
    if (!host_found[i]) {
      not_found_count++;
    }
  }
  EXPECT_EQ(not_found_count, key_num);

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例4: 混合场景 - 部分存在（带scores）
TEST_F(FindValueTest, MixedScenario_PartialExist_WithScores) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t insert_num = 1024;
  constexpr size_t query_num = 2048;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 插入 [1, 1024]
  vector<K> insert_keys;
  vector<V> insert_values;
  vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, insert_num, insert_keys, insert_values, insert_scores, true, stream);

  unordered_map<K, size_t> key_to_idx;
  for (size_t i = 0; i < insert_num; i++) {
    key_to_idx[insert_keys[i]] = i;
  }

  // 查询键: 前50%存在 [1, 1024]，后50%不存在 [10000, 11023]
  vector<K> query_keys(query_num);
  for (size_t i = 0; i < insert_num; i++) {
    query_keys[i] = insert_keys[i];
  }
  for (size_t i = insert_num; i < query_num; i++) {
    query_keys[i] = 10000 + (i - insert_num);
  }

  K* device_keys = nullptr;
  V* device_values = nullptr;
  S* device_scores = nullptr;
  bool* device_found = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        query_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        query_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                        query_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        query_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  ASSERT_EQ(aclrtMemcpy(device_keys, query_num * sizeof(K), query_keys.data(),
                        query_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  table.find(query_num, device_keys, device_values, device_found,
             device_scores, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 回收结果
  vector<V> result_values(query_num * dim);
  vector<S> result_scores(query_num);
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            query_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(result_values.data(), query_num * dim * sizeof(V),
                        device_values, query_num * dim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found, query_num * sizeof(bool), device_found,
                        query_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(result_scores.data(), query_num * sizeof(S),
                        device_scores, query_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  vector<bool> result_founds(host_found, host_found + query_num);
  VerifyFindResults<dim>(query_keys, result_values, result_founds,
                         insert_values, key_to_idx,
                         insert_num, query_num - insert_num,
                         &result_scores, &insert_scores);

  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}
