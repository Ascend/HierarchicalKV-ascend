/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <gtest/gtest.h>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace community_test_util;
using namespace npu::hkv;

namespace {

constexpr size_t kDim = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;

class FindTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kInitCapacity = 1024UL * 1024;
  static constexpr uint64_t kMaxCapacity = kInitCapacity;
  static constexpr uint64_t kKeyNum = kInitCapacity;

  void SetUp() override {
    init_env();
  }

  void verify_find_results(const std::vector<K>& host_keys,
                           const std::vector<V>& result_values,
                           const std::vector<K>& missed_keys,
                           const std::vector<int>& missed_indices,
                           int missed_size,
                           const std::vector<V>& expected_values,
                           size_t insert_num,
                           size_t dim) {
    std::unordered_map<K, size_t> key_to_insert_idx;
    for (size_t i = 0; i < insert_num; ++i) {
      key_to_insert_idx[host_keys[i]] = i;
    }

    std::vector<bool> founds(host_keys.size(), true);
    // Check missed.
    for (int j = 0; j < missed_size; ++j) {
      const int idx = missed_indices[j];
      ASSERT_GE(idx, 0) << "missed_indices[" << j << "] should be >= 0";
      ASSERT_LT(idx, static_cast<int>(host_keys.size()))
          << "missed_indices[" << j << "] should be within query range";
      EXPECT_EQ(host_keys[idx], missed_keys[j])
          << "missed key should match query key";
      founds[idx] = false;
    }

    size_t found_count = 0;
    // Check hit keys and returned vectors.
    for (size_t j = 0; j < host_keys.size(); ++j) {
      if (!founds[j]) {
        continue;
      }
      ++found_count;
      const auto it = key_to_insert_idx.find(host_keys[j]);
      ASSERT_NE(it, key_to_insert_idx.end())
          << "found key should exist in the inserted prefix";
      for (size_t k = 0; k < dim; ++k) {
        EXPECT_EQ(result_values[j * dim + k],
                  expected_values[it->second * dim + k])
            << "key=" << host_keys[j] << " dim_idx=" << k;
      }
    }

    EXPECT_EQ(found_count + static_cast<size_t>(missed_size), host_keys.size());
  }

  void test_find(size_t max_hbm_for_vectors_mb, size_t max_bucket_size,
                 double load_factor, bool pipeline_lookup,
                 int key_start = 0) {
    ASSERT_GE(load_factor, 0.0);
    ASSERT_LE(load_factor, 1.0);

    HashTableOptions options{};
    options.reserved_key_start_bit = key_start;
    options.init_capacity = kInitCapacity;
    options.max_capacity = kMaxCapacity;
    options.dim = kDim;
    options.max_hbm_for_vectors = max_hbm_for_vectors_mb * 1024UL * 1024UL;
    options.max_bucket_size = max_bucket_size;
    options.io_by_cpu = false;
    (void)pipeline_lookup;

    using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

    std::vector<K> host_keys(kKeyNum);
    std::vector<S> host_scores(kKeyNum);
    std::vector<V> host_vectors(kKeyNum * options.dim);
    create_random_keys<K, S, V, kDim>(host_keys.data(), host_scores.data(),
                                      host_vectors.data(), kKeyNum);

    K* d_keys = nullptr;
    S* d_scores = nullptr;
    V* d_vectors = nullptr;
    K* d_missed_keys = nullptr;
    int* d_missed_indices = nullptr;
    int* d_missed_size = nullptr;
    aclrtStream stream = nullptr;

    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_keys),
                          kKeyNum * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_scores),
                          kKeyNum * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_vectors),
                          kKeyNum * options.dim * sizeof(V),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_keys),
                          kKeyNum * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_indices),
                          kKeyNum * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_missed_size),
                          sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

    const size_t insert_num = static_cast<size_t>(kKeyNum * load_factor);
    ASSERT_EQ(aclrtMemcpy(d_keys, kKeyNum * sizeof(K), host_keys.data(),
                          kKeyNum * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(d_scores, kKeyNum * sizeof(S), host_scores.data(),
                          kKeyNum * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(d_vectors, kKeyNum * options.dim * sizeof(V),
                          host_vectors.data(),
                          kKeyNum * options.dim * sizeof(V),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    Table table;
    table.init(options);
    ASSERT_EQ(table.size(stream), 0UL);

    if (insert_num > 0) {
      table.insert_or_assign(insert_num, d_keys, d_vectors, d_scores, stream);
      ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
    }

    ASSERT_EQ(aclrtMemset(d_missed_size, sizeof(int), 0, sizeof(int)),
              ACL_ERROR_NONE);
    table.find(kKeyNum, d_keys, d_vectors, d_missed_keys, d_missed_indices,
               d_missed_size, d_scores, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    int missed_size = 0;
    ASSERT_EQ(aclrtMemcpy(&missed_size, sizeof(int), d_missed_size,
                          sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    if (insert_num == 0) {
      ASSERT_EQ(missed_size, static_cast<int>(kKeyNum));
    } else {
      ASSERT_GT(missed_size, 0);
      ASSERT_LT(missed_size, static_cast<int>(kKeyNum));
      ASSERT_EQ(missed_size,
                static_cast<int>(kKeyNum - table.size(stream)));

      std::vector<K> missed_keys(missed_size);
      std::vector<int> missed_indices(missed_size);
      std::vector<V> result_values(kKeyNum * options.dim);

      ASSERT_EQ(aclrtMemcpy(missed_keys.data(), missed_size * sizeof(K),
                            d_missed_keys, missed_size * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(missed_indices.data(), missed_size * sizeof(int),
                            d_missed_indices, missed_size * sizeof(int),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(result_values.data(),
                            kKeyNum * options.dim * sizeof(V), d_vectors,
                            kKeyNum * options.dim * sizeof(V),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);

      verify_find_results(host_keys, result_values, missed_keys,
                          missed_indices, missed_size, host_vectors,
                          insert_num, options.dim);
    }

    ASSERT_EQ(aclrtFree(d_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_scores), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_vectors), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_missed_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_missed_indices), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(d_missed_size), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }
};

TEST_F(FindTest, test_find_when_empty) {
  // pure HMEM
  test_find(0, 128, 0.0, true, 12);
  test_find(0, 256, 0.0, false);
  // hybrid
  test_find(32, 128, 0.0, true, 58);
  test_find(32, 256, 0.0, false);
  // pure HBM
  test_find(1024, 128, 0.0, true);
  test_find(1024, 256, 0.0, false, 12);
}

TEST_F(FindTest, test_find_when_full) {
  // pure HMEM
  test_find(0, 128, 1.0, true);
  test_find(0, 256, 1.0, false);
  // hybrid
  test_find(32, 128, 1.0, true);
  test_find(32, 256, 1.0, false, 60);
  // pure HBM
  test_find(1024, 128, 1.0, true);
  test_find(1024, 256, 1.0, false);
}

TEST_F(FindTest, test_find_load_factor) {
  // pure HMEM
  test_find(0, 128, 0.2, true, 45);
  test_find(0, 256, 0.2, false, 12);
  // hybrid
  test_find(32, 128, 0.2, true, 27);
  test_find(32, 256, 0.2, false, 53);
  // pure HBM
  test_find(1024, 128, 0.2, true, 9);
  test_find(1024, 256, 0.2, false, 38);

  // pure HMEM
  test_find(0, 128, 0.5, true, 21);
  test_find(0, 256, 0.5, false, 46);
  // hybrid
  test_find(32, 128, 0.5, true, 31);
  test_find(32, 256, 0.5, false, 59);
  // pure HBM
  test_find(1024, 128, 0.5, true, 4);
  test_find(1024, 256, 0.5, false, 22);

  // pure HMEM
  test_find(0, 128, 0.75, true, 11);
  test_find(0, 256, 0.75, false, 34);
  // hybrid
  test_find(32, 128, 0.75, true, 18);
  test_find(32, 256, 0.75, false, 47);
  // pure HBM
  test_find(1024, 128, 0.75, true, 7);
  test_find(1024, 256, 0.75, false, 29);
}

}  // namespace
