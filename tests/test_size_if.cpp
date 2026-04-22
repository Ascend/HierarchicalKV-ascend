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
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"
#include "test_device_data.h"
#include "../include/types.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

/**
 * @brief Predicate functor: counts keys with score < threshold
 * This matches the ExportIfPredFunctor in open-source HKV
 */
template <class K, class S>
struct ExportIfPredFunctor {
  __forceinline__ __simt_callee__ bool operator()(const K& key, const S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return score < threshold;
  }
};

/**
 * @brief Predicate functor: counts keys with score >= threshold
 */
template <class K, class S>
struct ScoreGtePredFunctor {
  __forceinline__ __simt_callee__ bool operator()(const K& key, const S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return score >= threshold;
  }
};

/**
 * @brief Predicate functor: counts keys with key & pattern == pattern
 */
template <class K, class S>
struct KeyMatchPredFunctor {
  __forceinline__ __simt_callee__ bool operator()(const K& key, const S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return (key & pattern) == pattern;
  }
};

// ============================================================================
// Test 1: size_if vs export_batch_if consistency test (matching open-source)
// ============================================================================
template <typename K, typename V, typename S>
void run_size_if_vs_export_test() {
  // Use get_table like other tests
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t capacity = 128UL * 1024;
  constexpr size_t key_num = 4 * 1024UL;

  init_env();

  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // Prepare device memory
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // Create data with 3 different score groups
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  vector<S> host_scores(key_num);

  // First third: score = 15
  // Second third: score = 30
  // Last third: score = 45
  for (size_t i = 0; i < key_num; i++) {
    host_keys[i] = static_cast<K>(i);
    if (i < key_num / 3) {
      host_scores[i] = 15;
    } else if (i < 2 * key_num / 3) {
      host_scores[i] = 30;
    } else {
      host_scores[i] = 45;
    }
    for (size_t j = 0; j < dim; j++) {
      host_values[i * dim + j] = static_cast<V>(i * dim + j);
    }
  }

  // Copy to device and insert
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  device_data.copy_scores(host_scores, key_num);

  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t* d_cnt = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_cnt), sizeof(size_t),
                        ACL_MEM_MALLOC_HUGE_FIRST), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_cnt, sizeof(size_t), 0, sizeof(size_t)), ACL_ERROR_NONE);

  // size_if: count keys with score < 40 (scores 15 and 30 should match)
  S threshold = 40;
  size_t h_cnt_size_if = 0;
  table->template size_if<ExportIfPredFunctor>(0, threshold, d_cnt,
                                                device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(&h_cnt_size_if, sizeof(size_t), d_cnt, sizeof(size_t),
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  // export_batch_if: export keys with score < 40
  ASSERT_EQ(aclrtMemset(d_cnt, sizeof(size_t), 0, sizeof(size_t)), ACL_ERROR_NONE);
  size_t* d_export_cnt = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_export_cnt), sizeof(size_t),
                        ACL_MEM_MALLOC_HUGE_FIRST), ACL_ERROR_NONE);

  K* d_export_keys = nullptr;
  V* d_export_values = nullptr;
  S* d_export_scores = nullptr;
  size_t scan_len = table->capacity();

  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_export_keys),
                        scan_len * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_export_values),
                        scan_len * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_export_scores),
                        scan_len * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  table->template export_batch_if<ExportIfPredFunctor>(
      0, threshold, scan_len, 0, d_export_cnt, d_export_keys,
      d_export_values, d_export_scores, device_data.stream);

  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  size_t h_cnt_export = 0;
  ASSERT_EQ(aclrtMemcpy(&h_cnt_export, sizeof(size_t), d_export_cnt,
                        sizeof(size_t), ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  // Verify both return the same count
  ASSERT_EQ(h_cnt_size_if, h_cnt_export)
      << "size_if: " << h_cnt_size_if << ", export_batch_if: " << h_cnt_export;

  // Cleanup
  ASSERT_EQ(aclrtFree(d_cnt), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_export_cnt), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_export_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_export_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(d_export_scores), ACL_ERROR_NONE);
}

// ============================================================================
// Test 2: Empty table test
// ============================================================================
template <typename K, typename V, typename S>
void run_size_if_empty_test() {
  init_env();

  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t capacity = 1024;

  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);

  size_t* d_cnt = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_cnt), sizeof(size_t),
                        ACL_MEM_MALLOC_HUGE_FIRST), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_cnt, sizeof(size_t), 0, sizeof(size_t)), ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  size_t h_cnt = 0;
  table->template size_if<ExportIfPredFunctor>(0, 100, d_cnt, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(&h_cnt, sizeof(size_t), d_cnt, sizeof(size_t),
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  ASSERT_EQ(h_cnt, 0);

  ASSERT_EQ(aclrtFree(d_cnt), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// ============================================================================
// Test 3: All match test
// ============================================================================
template <typename K, typename V, typename S>
void run_size_if_all_match_test() {
  init_env();

  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t capacity = 128 * 1024;
  constexpr size_t key_num = 4 * 1024UL;

  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);

  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  vector<S> host_scores(key_num);

  for (size_t i = 0; i < key_num; i++) {
    host_keys[i] = static_cast<K>(i);
    host_scores[i] = static_cast<S>(i + 1);  // scores: 1, 2, 3, ...
    for (size_t j = 0; j < dim; j++) {
      host_values[i * dim + j] = static_cast<V>(i * dim + j);
    }
  }

  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  device_data.copy_scores(host_scores, key_num);

  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t* d_cnt = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_cnt), sizeof(size_t),
                        ACL_MEM_MALLOC_HUGE_FIRST), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_cnt, sizeof(size_t), 0, sizeof(size_t)), ACL_ERROR_NONE);

  // Use ScoreGtePredFunctor with threshold=1, all scores >= 1 should match
  size_t h_cnt = 0;
  table->template size_if<ScoreGtePredFunctor>(0, static_cast<S>(1), d_cnt,
                                                device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(&h_cnt, sizeof(size_t), d_cnt, sizeof(size_t),
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  ASSERT_EQ(h_cnt, key_num);

  ASSERT_EQ(aclrtFree(d_cnt), ACL_ERROR_NONE);
}

// ============================================================================
// Test 4: No match test
// ============================================================================
template <typename K, typename V, typename S>
void run_size_if_no_match_test() {
  init_env();

  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t capacity = 128 * 1024;
  constexpr size_t key_num = 4 * 1024UL;

  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);

  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  vector<S> host_scores(key_num);

  for (size_t i = 0; i < key_num; i++) {
    host_keys[i] = static_cast<K>(i);
    host_scores[i] = static_cast<S>(i + 1);
    for (size_t j = 0; j < dim; j++) {
      host_values[i * dim + j] = static_cast<V>(i * dim + j);
    }
  }

  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  device_data.copy_scores(host_scores, key_num);

  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t* d_cnt = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_cnt), sizeof(size_t),
                        ACL_MEM_MALLOC_HUGE_FIRST), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_cnt, sizeof(size_t), 0, sizeof(size_t)), ACL_ERROR_NONE);

  // Use ScoreGtePredFunctor with threshold > max score, no match
  size_t h_cnt = 0;
  table->template size_if<ScoreGtePredFunctor>(0, static_cast<S>(key_num + 100),
                                                d_cnt, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(&h_cnt, sizeof(size_t), d_cnt, sizeof(size_t),
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  ASSERT_EQ(h_cnt, 0);

  ASSERT_EQ(aclrtFree(d_cnt), ACL_ERROR_NONE);
}

// ============================================================================
// Test 5: Key pattern matching test
// ============================================================================
template <typename K, typename V, typename S>
void run_size_if_key_pattern_test() {
  init_env();

  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t capacity = 128 * 1024;
  constexpr size_t key_num = 4 * 1024UL;

  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);

  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  vector<S> host_scores(key_num);

  for (size_t i = 0; i < key_num; i++) {
    host_keys[i] = static_cast<K>(i);
    host_scores[i] = static_cast<S>(i + 1);
    for (size_t j = 0; j < dim; j++) {
      host_values[i * dim + j] = static_cast<V>(i * dim + j);
    }
  }

  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  device_data.copy_scores(host_scores, key_num);

  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t* d_cnt = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_cnt), sizeof(size_t),
                        ACL_MEM_MALLOC_HUGE_FIRST), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_cnt, sizeof(size_t), 0, sizeof(size_t)), ACL_ERROR_NONE);

  // Test: count keys where (key & 0xF) == 0xF
  K pattern = 0xF;
  size_t h_cnt = 0;
  table->template size_if<KeyMatchPredFunctor>(pattern, 0, d_cnt,
                                                device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(&h_cnt, sizeof(size_t), d_cnt, sizeof(size_t),
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  // Expected: keys 15, 31, 47, 63, ..., 4095
  size_t expected = 0;
  for (size_t i = 0; i < key_num; i++) {
    if ((i & 0xF) == 0xF) {
      expected++;
    }
  }
  ASSERT_EQ(h_cnt, expected);

  ASSERT_EQ(aclrtFree(d_cnt), ACL_ERROR_NONE);
}

// ============================================================================
// Test 6: Large dataset test
// ============================================================================
template <typename K, typename V, typename S>
void run_size_if_large_test() {
  init_env();

  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t capacity = 128UL * 1024;
  constexpr size_t key_num = 64UL * 1024;

  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);

  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  vector<S> host_scores(key_num);

  for (size_t i = 0; i < key_num; i++) {
    host_keys[i] = static_cast<K>(i);
    host_scores[i] = static_cast<S>((i % 100) + 1);  // scores cycle 1-100
    for (size_t j = 0; j < dim; j++) {
      host_values[i * dim + j] = static_cast<V>(i * dim + j);
    }
  }

  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  device_data.copy_scores(host_scores, key_num);

  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  size_t* d_cnt = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&d_cnt), sizeof(size_t),
                        ACL_MEM_MALLOC_HUGE_FIRST), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(d_cnt, sizeof(size_t), 0, sizeof(size_t)), ACL_ERROR_NONE);

  // Use ScoreGtePredFunctor with threshold=50, scores 50-100 should match
  size_t h_cnt = 0;
  table->template size_if<ScoreGtePredFunctor>(0, static_cast<S>(50), d_cnt,
                                                device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(&h_cnt, sizeof(size_t), d_cnt, sizeof(size_t),
                        ACL_MEMCPY_DEVICE_TO_HOST), ACL_ERROR_NONE);

  // Expected: scores 50-100 (51 values), each repeats
  size_t expected = (key_num / 100) * 51 + ((key_num % 100) >= 50 ? (key_num % 100) - 50 + 1 : 0);
  ASSERT_EQ(h_cnt, expected);

  ASSERT_EQ(aclrtFree(d_cnt), ACL_ERROR_NONE);
}

// ============================================================================
// Test cases
// ============================================================================

// Test 1: size_if vs export_batch_if consistency
TEST(test_size_if, test_size_if_vs_export) {
  run_size_if_vs_export_test<uint64_t, float, uint64_t>();
}

// Test 2: Empty table
TEST(test_size_if, test_size_if_empty) {
  run_size_if_empty_test<uint64_t, float, uint64_t>();
}

// Test 3: All match
TEST(test_size_if, test_size_if_all_match) {
  run_size_if_all_match_test<uint64_t, float, uint64_t>();
}

// Test 4: No match
TEST(test_size_if, test_size_if_no_match) {
  run_size_if_no_match_test<uint64_t, float, uint64_t>();
}

// Test 5: Key pattern
TEST(test_size_if, test_size_if_key_pattern) {
  run_size_if_key_pattern_test<uint64_t, float, uint64_t>();
}

// Test 6: Large dataset
TEST(test_size_if, test_size_if_large) {
  run_size_if_large_test<uint64_t, float, uint64_t>();
}