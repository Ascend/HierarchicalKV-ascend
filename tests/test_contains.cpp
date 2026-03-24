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
#include <vector>
#include <unordered_set>
#include "test_find_base.h"

class ContainsTest : public FindTestBase {
 public:
  void VerifyContainsResults(const vector<K>& query_keys,
                             const vector<bool>& result_founds,
                             const unordered_set<K>& existing_keys,
                             size_t expected_found_num,
                             size_t expected_not_found_num) {
    size_t found_num = 0;
    size_t not_found_num = 0;
    for (size_t i = 0; i < query_keys.size(); i++) {
      bool should_exist = existing_keys.count(query_keys[i]) > 0;
      if (should_exist) {
        EXPECT_TRUE(result_founds[i]) << "key=" << query_keys[i] << " should be found";
        if (result_founds[i]) {
          found_num++;
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

  template<typename TableType>
  void RunContains(TableType& table,
                   const vector<K>& query_keys,
                   vector<bool>& result_founds,
                   aclrtStream stream) {
    size_t query_num = query_keys.size();
    K* device_keys = nullptr;
    bool* device_found = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          query_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                          query_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);

    ASSERT_EQ(aclrtMemcpy(device_keys, query_num * sizeof(K), query_keys.data(),
                          query_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    table.contains(query_num, device_keys, device_found, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    bool* host_found = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                              query_num * sizeof(bool)),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(host_found, query_num * sizeof(bool), device_found,
                          query_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    result_founds.assign(host_found, host_found + query_num);

    ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  }
};

// 用例1: 边界测试 - 空查询 (n=0)
TEST_F(ContainsTest, BoundaryTest_EmptyQuery) {
  constexpr size_t dim = DEFAULT_DIM;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  table.contains(0, nullptr, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  SUCCEED();
}

// 用例2: 基本功能 - 小规模全存在
TEST_F(ContainsTest, BasicFunction_SmallScaleAllExist) {
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

  unordered_set<K> existing_keys(host_keys.begin(), host_keys.end());

  vector<bool> result_founds;
  RunContains(table, host_keys, result_founds, stream);

  VerifyContainsResults(host_keys, result_founds, existing_keys, key_num, 0);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例3: 基本功能 - 小规模全不存在
TEST_F(ContainsTest, BasicFunction_SmallScaleNoneExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> insert_keys;
  vector<V> insert_values;
  vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, key_num, insert_keys, insert_values, insert_scores, false, stream);

  // 查询键 [100000, 101023]，全部不存在
  vector<K> query_keys(key_num);
  create_continuous_keys<K, S, V, dim>(query_keys.data(), nullptr,
                                       nullptr, key_num, 100000);

  unordered_set<K> existing_keys(insert_keys.begin(), insert_keys.end());

  vector<bool> result_founds;
  RunContains(table, query_keys, result_founds, stream);

  VerifyContainsResults(query_keys, result_founds, existing_keys, 0, key_num);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例4: 混合场景 - 部分存在
TEST_F(ContainsTest, MixedScenario_PartialExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t insert_num = 1024;
  constexpr size_t query_num = 2048;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> insert_keys;
  vector<V> insert_values;
  vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, insert_num, insert_keys, insert_values, insert_scores, true, stream);

  unordered_set<K> existing_keys(insert_keys.begin(), insert_keys.end());

  // 前50%存在 [1, 1024]，后50%不存在 [10000, 11023]
  vector<K> query_keys(query_num);
  for (size_t i = 0; i < insert_num; i++) {
    query_keys[i] = insert_keys[i];
  }
  for (size_t i = insert_num; i < query_num; i++) {
    query_keys[i] = 10000 + (i - insert_num);
  }

  vector<bool> result_founds;
  RunContains(table, query_keys, result_founds, stream);

  VerifyContainsResults(query_keys, result_founds, existing_keys,
                        insert_num, query_num - insert_num);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例5: 中等规模测试 - 部分存在
TEST_F(ContainsTest, MediumScale_PartialExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t insert_num = 5 * 1024;
  constexpr size_t query_num = 10 * 1024;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> insert_keys;
  vector<V> insert_values;
  vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, insert_num, insert_keys, insert_values, insert_scores, true, stream);

  unordered_set<K> existing_keys(insert_keys.begin(), insert_keys.end());

  // 前50%存在，后50%不存在
  vector<K> query_keys(query_num);
  for (size_t i = 0; i < insert_num; i++) {
    query_keys[i] = insert_keys[i];
  }
  for (size_t i = insert_num; i < query_num; i++) {
    query_keys[i] = 20000 + (i - insert_num);
  }

  vector<bool> result_founds;
  RunContains(table, query_keys, result_founds, stream);

  VerifyContainsResults(query_keys, result_founds, existing_keys,
                        insert_num, query_num - insert_num);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例6: 大规模测试 - 64K全存在
TEST_F(ContainsTest, LargeScale_AllExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 64 * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> host_keys;
  vector<V> host_values;
  vector<S> host_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, key_num, host_keys, host_values, host_scores, false, stream);

  unordered_set<K> existing_keys(host_keys.begin(), host_keys.end());

  vector<bool> result_founds;
  RunContains(table, host_keys, result_founds, stream);

  VerifyContainsResults(host_keys, result_founds, existing_keys, key_num, 0);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例7: 不同dim测试 - dim=128
TEST_F(ContainsTest, DifferentDim_128) {
  constexpr size_t dim = 128;
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

  unordered_set<K> existing_keys(host_keys.begin(), host_keys.end());

  vector<bool> result_founds;
  RunContains(table, host_keys, result_founds, stream);

  VerifyContainsResults(host_keys, result_founds, existing_keys, key_num, 0);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例8: 不同dim测试 - dim=4（小dim）
TEST_F(ContainsTest, DifferentDim_4) {
  constexpr size_t dim = 4;
  constexpr size_t key_num = 2048;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> host_keys;
  vector<V> host_values;
  vector<S> host_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, key_num, host_keys, host_values, host_scores, false, stream);

  unordered_set<K> existing_keys(host_keys.begin(), host_keys.end());

  vector<bool> result_founds;
  RunContains(table, host_keys, result_founds, stream);

  VerifyContainsResults(host_keys, result_founds, existing_keys, key_num, 0);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例9: 重复键测试 - 查询中包含重复key
TEST_F(ContainsTest, DuplicateKeys) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t unique_key_num = 200;
  constexpr size_t query_num = 1000;

  HashTable<K, V, S, EvictStrategy::kCustomized> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> insert_keys;
  vector<V> insert_values;
  vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, unique_key_num, insert_keys, insert_values, insert_scores,
      true, stream);

  unordered_set<K> existing_keys(insert_keys.begin(), insert_keys.end());

  // 创建含重复键的查询数组：[1,2,...,200,1,2,...,200,...]
  vector<K> query_keys(query_num);
  for (size_t i = 0; i < query_num; i++) {
    query_keys[i] = insert_keys[i % unique_key_num];
  }

  vector<bool> result_founds;
  RunContains(table, query_keys, result_founds, stream);

  VerifyContainsResults(query_keys, result_founds, existing_keys, query_num, 0);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例10: 空表查询
TEST_F(ContainsTest, EmptyTable_NoneExist) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t query_num = 100;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  vector<K> query_keys(query_num);
  create_continuous_keys<K, S, V, dim>(query_keys.data(), nullptr,
                                       nullptr, query_num, 1);

  unordered_set<K> existing_keys;

  vector<bool> result_founds;
  RunContains(table, query_keys, result_founds, stream);

  VerifyContainsResults(query_keys, result_founds, existing_keys, 0, query_num);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例11: IS_RESERVED_KEY 保留键场景 - 混合正常key和所有保留key
TEST_F(ContainsTest, ReservedKeys_MixedWithNormalKeys) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t normal_key_num = 1024;

  const vector<K> reserved_keys = {
      UINT64_C(0xFFFFFFFFFFFFFFFC),  // RESERVED_KEY_MASK
      UINT64_C(0xFFFFFFFFFFFFFFFD),  // LOCKED_KEY
      UINT64_C(0xFFFFFFFFFFFFFFFE),  // RECLAIM_KEY
      UINT64_C(0xFFFFFFFFFFFFFFFF),  // EMPTY_KEY
  };
  const size_t reserved_num = reserved_keys.size();
  const size_t query_num = normal_key_num + reserved_num;

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 插入正常键 [1, 1024]
  vector<K> insert_keys;
  vector<V> insert_values;
  vector<S> insert_scores;
  InsertContinuousKeys<decltype(table), dim>(
      table, normal_key_num, insert_keys, insert_values, insert_scores,
      false, stream);

  unordered_set<K> existing_keys(insert_keys.begin(), insert_keys.end());

  // 查询键 = 正常键 + 保留键
  vector<K> query_keys(query_num);
  for (size_t i = 0; i < normal_key_num; i++) {
    query_keys[i] = insert_keys[i];
  }
  for (size_t i = 0; i < reserved_num; i++) {
    query_keys[normal_key_num + i] = reserved_keys[i];
  }

  vector<bool> result_founds;
  RunContains(table, query_keys, result_founds, stream);

  // 正常键全部找到
  size_t found_count = 0;
  for (size_t i = 0; i < normal_key_num; i++) {
    EXPECT_TRUE(result_founds[i]) << "normal key=" << query_keys[i] << " should be found";
    if (result_founds[i]) {
      found_count++;
    }
  }
  EXPECT_EQ(found_count, normal_key_num);

  // 保留键全部未找到
  size_t reserved_not_found_count = 0;
  for (size_t i = 0; i < reserved_num; i++) {
    size_t idx = normal_key_num + i;
    EXPECT_FALSE(result_founds[idx])
        << "reserved key=0x" << std::hex << query_keys[idx]
        << " should NOT be found";
    if (!result_founds[idx]) {
      reserved_not_found_count++;
    }
  }
  EXPECT_EQ(reserved_not_found_count, reserved_num);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}
