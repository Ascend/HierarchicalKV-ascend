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

// find(miss) 专用测试类
class FindMissTest : public FindTestBase {
 public:
  static constexpr uint64_t INIT_CAPACITY = 1024UL * 1024;
  static constexpr uint64_t KEY_NUM = INIT_CAPACITY;
  static constexpr size_t DIM = 16;

  void VerifyFindMissResults(const vector<K>& query_keys,
                             const vector<V>& result_values,
                             const vector<K>& missed_keys,
                             const vector<int>& missed_indices,
                             int missed_size,
                             const vector<V>& expected_values,
                             const unordered_map<K, size_t>& key_to_insert_idx,
                             size_t dim) {
    vector<bool> founds(query_keys.size(), true);

    for (int j = 0; j < missed_size; ++j) {
      int idx = missed_indices[j];
      ASSERT_GE(idx, 0) << "missed_indices[" << j << "] should >= 0";
      ASSERT_LT(idx, static_cast<int>(query_keys.size()))
          << "missed_indices[" << j << "] should < query size";
      EXPECT_EQ(query_keys[idx], missed_keys[j])
          << "missed_keys[" << j << "] should match query_keys[" << idx << "]";
      founds[idx] = false;
    }

    size_t found_count = 0;
    for (size_t j = 0; j < query_keys.size(); ++j) {
      if (founds[j]) {
        found_count++;
        auto it = key_to_insert_idx.find(query_keys[j]);
        ASSERT_NE(it, key_to_insert_idx.end())
            << "found key=" << query_keys[j] << " should exist in insert set";
        size_t insert_idx = it->second;
        for (size_t k = 0; k < dim; ++k) {
          EXPECT_EQ(result_values[j * dim + k],
                    expected_values[insert_idx * dim + k])
              << "key=" << query_keys[j] << " dim_idx=" << k;
        }
      }
    }

    EXPECT_EQ(found_count + static_cast<size_t>(missed_size),
              query_keys.size());
  }

  void TestFindMiss(size_t max_hbm_for_vectors_mb, size_t max_bucket_size,
                    double load_factor, int key_start = 0) {
    ASSERT_TRUE(load_factor >= 0.0 && load_factor <= 1.0)
        << "Invalid load_factor";

    constexpr size_t dim = DIM;

    HashTableOptions options;
    options.init_capacity = INIT_CAPACITY;
    options.max_capacity = INIT_CAPACITY;
    options.dim = dim;
    options.max_hbm_for_vectors = max_hbm_for_vectors_mb * 1024UL * 1024UL;
    options.max_bucket_size = max_bucket_size;
    options.reserved_key_start_bit = key_start;

    using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;
    Table table;
    table.init(options);

    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

    size_t insert_num = static_cast<size_t>(KEY_NUM * load_factor);

    vector<K> host_keys(KEY_NUM);
    vector<S> host_scores(KEY_NUM);
    vector<V> host_values(KEY_NUM * dim);
    create_continuous_keys<K, S, V, dim>(
        host_keys.data(), host_scores.data(), host_values.data(),
        KEY_NUM, 1);

    K* d_keys = nullptr;
    S* d_scores = nullptr;
    V* d_values = nullptr;
    K* d_missed_keys = nullptr;
    int* d_missed_indices = nullptr;
    int* d_missed_size = nullptr;

    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_keys),
                          KEY_NUM * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_scores),
                          KEY_NUM * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_values),
                          KEY_NUM * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_keys),
                          KEY_NUM * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_indices),
                          KEY_NUM * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_size),
                          sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);

    if (insert_num > 0) {
      ASSERT_EQ(aclrtMemcpy(d_keys, KEY_NUM * sizeof(K), host_keys.data(),
                            insert_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(d_scores, KEY_NUM * sizeof(S), host_scores.data(),
                            insert_num * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(d_values, KEY_NUM * dim * sizeof(V),
                            host_values.data(), insert_num * dim * sizeof(V),
                            ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);

      table.insert_or_assign(insert_num, d_keys, d_values, d_scores, stream);
      ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
    }

    ASSERT_EQ(aclrtMemcpy(d_keys, KEY_NUM * sizeof(K), host_keys.data(),
                          KEY_NUM * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    ASSERT_EQ(aclrtMemset(d_missed_size, sizeof(int), 0, sizeof(int)),
              ACL_ERROR_NONE);

    table.find(KEY_NUM, d_keys, d_values, d_missed_keys, d_missed_indices,
               d_missed_size, d_scores, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    int h_missed_size = 0;
    ASSERT_EQ(aclrtMemcpy(&h_missed_size, sizeof(int), d_missed_size,
                          sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    if (insert_num == 0) {
      ASSERT_EQ(h_missed_size, static_cast<int>(KEY_NUM));
    } else {
      ASSERT_GE(h_missed_size, 0);
      auto expected_missed_size = KEY_NUM - table.size(stream);
      ASSERT_EQ(h_missed_size, expected_missed_size);
      ASSERT_LT(h_missed_size, static_cast<int>(KEY_NUM));

      K* h_missed_keys = nullptr;
      int* h_missed_indices = nullptr;
      ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&h_missed_keys),
                                KEY_NUM * sizeof(K)),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&h_missed_indices),
                                KEY_NUM * sizeof(int)),
                ACL_ERROR_NONE);

      ASSERT_EQ(aclrtMemcpy(h_missed_keys, h_missed_size * sizeof(K),
                            d_missed_keys, h_missed_size * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(h_missed_indices, h_missed_size * sizeof(int),
                            d_missed_indices, h_missed_size * sizeof(int),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);

      vector<V> result_values(KEY_NUM * dim);
      ASSERT_EQ(aclrtMemcpy(result_values.data(), KEY_NUM * dim * sizeof(V),
                            d_values, KEY_NUM * dim * sizeof(V),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);

      unordered_map<K, size_t> key_to_idx;
      for (size_t i = 0; i < insert_num; i++) {
        key_to_idx[host_keys[i]] = i;
      }

      vector<K> missed_keys_vec(h_missed_keys, h_missed_keys + h_missed_size);
      vector<int> missed_indices_vec(h_missed_indices,
                                     h_missed_indices + h_missed_size);

      VerifyFindMissResults(host_keys, result_values,
                            missed_keys_vec, missed_indices_vec,
                            h_missed_size,
                            host_values, key_to_idx, dim);

      ASSERT_EQ(aclrtFreeHost(h_missed_keys), ACL_ERROR_NONE);
      ASSERT_EQ(aclrtFreeHost(h_missed_indices), ACL_ERROR_NONE);
    }

    ASSERT_EQ(aclrtFree(d_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_scores), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_values), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_missed_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_missed_indices), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_missed_size), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }
};

// 用例1: 空表测试 - load_factor=0.0，所有key都应miss
TEST_F(FindMissTest, WhenEmpty) {
  // pure HBM
  TestFindMiss(1024, 128, 0.0);
  TestFindMiss(1024, 256, 0.0, 12);

  // pure HMEM
  TestFindMiss(0, 128, 0.0, 12);
  TestFindMiss(0, 256, 0.0);

  // hybrid
  TestFindMiss(32, 128, 0.0, 58);
  TestFindMiss(32, 256, 0.0);
}

// 用例2: 满表测试 - load_factor=1.0，所有key都应找到
TEST_F(FindMissTest, WhenFull) {
  // pure HBM
  TestFindMiss(1024, 128, 1.0);
  TestFindMiss(1024, 256, 1.0);

  // pure HMEM
  TestFindMiss(0, 128, 1.0);
  TestFindMiss(0, 256, 1.0);

  // hybrid
  TestFindMiss(32, 128, 1.0);
  TestFindMiss(32, 256, 1.0, 60);
}

// 用例3: 不同load_factor - 部分命中
TEST_F(FindMissTest, LoadFactor) {
  // load_factor = 0.2
  // pure HBM
  TestFindMiss(1024, 128, 0.2, 9);
  TestFindMiss(1024, 256, 0.2, 38);
  // pure HMEM
  TestFindMiss(0, 128, 0.2, 45);
  TestFindMiss(0, 256, 0.2, 12);
  // hybrid
  TestFindMiss(32, 128, 0.2, 27);
  TestFindMiss(32, 256, 0.2, 53);

  // load_factor = 0.5
  // pure HBM
  TestFindMiss(1024, 128, 0.5, 4);
  TestFindMiss(1024, 256, 0.5, 22);
  // pure HMEM
  TestFindMiss(0, 128, 0.5, 21);
  TestFindMiss(0, 256, 0.5, 46);
  // hybrid
  TestFindMiss(32, 128, 0.5, 31);
  TestFindMiss(32, 256, 0.5, 59);

  // load_factor = 0.75
  // pure HBM
  TestFindMiss(1024, 128, 0.75, 7);
  TestFindMiss(1024, 256, 0.75, 29);
  // pure HMEM
  TestFindMiss(0, 128, 0.75, 11);
  TestFindMiss(0, 256, 0.75, 34);
  // hybrid
  TestFindMiss(32, 128, 0.75, 18);
  TestFindMiss(32, 256, 0.75, 47);
}
