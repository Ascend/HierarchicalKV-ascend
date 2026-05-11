/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace community_test_util;

namespace {

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = HashTableOptions;

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

template <class KType, class SType>
struct EraseIfPredFunctor {
  __forceinline__ __simt_callee__ bool operator()(const KType& key,
                                                  SType& score,
                                                  const KType& pattern,
                                                  const SType& threshold) {
    return (((key & 0x7f) > pattern) && (score > threshold));
  }
};

template <class KType, class SType>
struct ExportIfPredFunctor {
  __forceinline__ __simt_callee__ bool operator()(const KType& key,
                                                  SType& score,
                                                  const KType& pattern,
                                                  const SType& threshold) {
    (void)key;
    (void)pattern;
    return score > threshold;
  }
};

template <class T>
struct DeviceArray {
  T* ptr = nullptr;
  size_t count = 0;

  ~DeviceArray() {
    if (ptr != nullptr) {
      aclrtFree(ptr);
    }
  }

  T* get() const { return ptr; }

  void alloc(size_t n) {
    count = n;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * count,
                          ACL_MEM_MALLOC_HUGE_FIRST));
  }

  void memset(int value, aclrtStream stream) {
    ACL_CHECK(aclrtMemsetAsync(ptr, sizeof(T) * count, value,
                               sizeof(T) * count, stream));
  }

  void copy_from_host(const std::vector<T>& host, aclrtStream stream) {
    ASSERT_LE(host.size(), count);
    ACL_CHECK(aclrtMemcpyAsync(ptr, sizeof(T) * count, host.data(),
                               sizeof(T) * host.size(),
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
  }

  void copy_to_host(std::vector<T>* host, size_t n, aclrtStream stream) {
    if (n == 0) {
      return;
    }
    ASSERT_LE(n, count);
    host->assign(n, T{});
    ACL_CHECK(aclrtMemcpyAsync(host->data(), sizeof(T) * n, ptr, sizeof(T) * n,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }
};

struct StreamGuard {
  aclrtStream stream = nullptr;

  ~StreamGuard() {
    if (stream != nullptr) {
      aclrtDestroyStream(stream);
    }
  }

  void create() { ACL_CHECK(aclrtCreateStream(&stream)); }
};

TableOptions make_options(size_t init_capacity, size_t max_capacity,
                         size_t max_hbm_for_vectors, int key_start = 0,
                         size_t max_bucket_size = 128) {
  set_simd_value_move_enabled(max_hbm_for_vectors == 0);
  TableOptions options{};
  options.reserved_key_start_bit = key_start;
  options.init_capacity = init_capacity;
  options.max_capacity = max_capacity;
  options.dim = DIM;
  options.max_bucket_size = max_bucket_size;
  options.max_hbm_for_vectors = GB(max_hbm_for_vectors);
  return options;
}

template <int Strategy>
S* scores_for_insert(S* d_scores) {
  if constexpr (Strategy == EvictStrategy::kLru ||
                Strategy == EvictStrategy::kEpochLru) {
    return nullptr;
  }
  return d_scores;
}

template <int Strategy>
void set_epoch_if_needed(HashTable<K, V, S, Strategy>* table, S epoch) {
  if constexpr (Strategy == EvictStrategy::kEpochLru ||
                Strategy == EvictStrategy::kEpochLfu) {
    table->set_global_epoch(epoch);
  }
}

template <class Table, class VType>
void find_or_insert_safe(Table* table, uint64_t key_num, K* d_keys,
                         S* d_scores, VType* d_vectors, uint64_t dim,
                         aclrtStream stream, bool unique_key = true) {
  (void)dim;
  table->find_or_insert(key_num, d_keys, d_vectors, d_scores, stream,
                        unique_key);
  ACL_CHECK(aclrtSynchronizeStream(stream));
}

template <class Table>
void find_into_contiguous(Table* table, size_t key_num, K* d_keys, V* d_vectors,
                        bool* d_found, S* d_scores, aclrtStream stream) {
  DeviceArray<V*> d_vectors_ptr;
  d_vectors_ptr.alloc(key_num);
  d_vectors_ptr.memset(0, stream);
  table->find(key_num, d_keys, d_vectors_ptr.get(), d_found, d_scores, stream);
  read_from_ptr(d_vectors_ptr.get(), d_vectors, DIM, key_num, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
}

void verify_found_values_and_scores(const std::vector<K>& keys,
                                const std::vector<S>& scores,
                                const std::vector<V>& values,
                                const std::vector<uint8_t>& found,
                                size_t expected_found,
                                bool verify_score = true) {
  size_t found_num = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!found[i]) {
      continue;
    }
    ++found_num;
    if (verify_score) {
      ASSERT_EQ(scores[i], keys[i]);
    }
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(values[i * DIM + j], static_cast<float>(keys[i] * 0.00001));
    }
  }
  ASSERT_EQ(found_num, expected_found);
}

void copy_bool_to_host(bool* device, std::vector<uint8_t>* host, size_t n,
                    aclrtStream stream) {
  host->assign(n, uint8_t{0});
  ACL_CHECK(aclrtMemcpyAsync(host->data(), sizeof(uint8_t) * n, device,
                             sizeof(bool) * n, ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
}

template <class Table>
void export_and_verify_all(Table* table, size_t expected_count, K* d_keys,
                        V* d_vectors, S* d_scores, aclrtStream stream,
                        bool verify_score = true) {
  const size_t exported =
      table->export_batch(table->capacity(), 0, d_keys, d_vectors, d_scores,
                          stream);
  ASSERT_EQ(exported, expected_count);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::vector<K> h_keys;
  std::vector<V> h_vectors;
  std::vector<S> h_scores;
  DeviceArray<K> key_view;
  DeviceArray<V> value_view;
  DeviceArray<S> score_view;
  key_view.ptr = d_keys;
  key_view.count = expected_count;
  value_view.ptr = d_vectors;
  value_view.count = expected_count * DIM;
  score_view.ptr = d_scores;
  score_view.count = expected_count;
  key_view.copy_to_host(&h_keys, expected_count, stream);
  value_view.copy_to_host(&h_vectors, expected_count * DIM, stream);
  score_view.copy_to_host(&h_scores, expected_count, stream);
  key_view.ptr = nullptr;
  value_view.ptr = nullptr;
  score_view.ptr = nullptr;

  for (size_t i = 0; i < expected_count; ++i) {
    if (verify_score) {
      ASSERT_EQ(h_scores[i], h_keys[i]);
    }
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(h_vectors[i * DIM + j],
                static_cast<float>(h_keys[i] * 0.00001));
    }
  }
}

template <typename T, std::size_t N>
std::array<T, N> range(const T start) {
  std::array<T, N> result{};
  for (size_t i = 0; i < N; ++i) {
    result[i] = start + static_cast<T>(i);
  }
  return result;
}

template <class Table>
void export_to_host(Table* table, size_t expected_count, K* d_keys, V* d_vectors,
                  S* d_scores, std::vector<K>* h_keys,
                  std::vector<V>* h_vectors, std::vector<S>* h_scores,
                  aclrtStream stream) {
  const size_t exported =
      table->export_batch(table->capacity(), 0, d_keys, d_vectors, d_scores,
                          stream);
  ASSERT_EQ(exported, expected_count);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  DeviceArray<K> key_view;
  DeviceArray<V> value_view;
  DeviceArray<S> score_view;
  key_view.ptr = d_keys;
  key_view.count = expected_count;
  value_view.ptr = d_vectors;
  value_view.count = expected_count * DIM;
  score_view.ptr = d_scores;
  score_view.count = expected_count;
  key_view.copy_to_host(h_keys, expected_count, stream);
  value_view.copy_to_host(h_vectors, expected_count * DIM, stream);
  score_view.copy_to_host(h_scores, expected_count, stream);
  key_view.ptr = nullptr;
  value_view.ptr = nullptr;
  score_view.ptr = nullptr;
}

void verify_exported_values(const std::vector<K>& keys,
                          const std::vector<V>& values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(values[i * DIM + j],
                static_cast<float>(keys[i] * 0.00001));
    }
  }
}

void test_basic(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, MAX_CAPACITY, max_hbm_for_vectors, key_start);
  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(),
                                   h_vectors.data(), KEY_NUM);

  StreamGuard guard;
  guard.create();
  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<V> d_new_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_new_vectors.alloc(KEY_NUM * DIM);
  d_found.alloc(KEY_NUM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.copy_from_host(h_vectors, guard.stream);

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);

  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  d_found.memset(0, guard.stream);
  d_scores.memset(0, guard.stream);
  d_vectors.memset(0, guard.stream);
  find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_found.get(), d_scores.get(), guard.stream);
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  d_scores.copy_to_host(&h_scores, KEY_NUM, guard.stream);
  d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);
  verify_found_values_and_scores(h_keys, h_scores, h_vectors, h_found, KEY_NUM);

  d_new_vectors.memset(2, guard.stream);
  table->assign(KEY_NUM, d_keys.get(), d_new_vectors.get(), d_scores.get(),
                guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  d_found.memset(0, guard.stream);
  d_new_vectors.memset(0, guard.stream);
  find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_new_vectors.get(),
                     d_found.get(), d_scores.get(), guard.stream);
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  d_new_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);
  const uint32_t i_value = 0x2020202;
  const float expected_memset_value =
      *(reinterpret_cast<const float*>(&i_value));
  size_t found_num = 0;
  for (size_t i = 0; i < KEY_NUM; ++i) {
    if (h_found[i]) {
      ++found_num;
    }
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(h_vectors[i * DIM + j], expected_memset_value);
    }
  }
  ASSERT_EQ(found_num, KEY_NUM);

  table->accum_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(),
                         d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  table->erase(KEY_NUM >> 1, d_keys.get(), guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM >> 1);

  table->clear(guard.stream);
  ASSERT_EQ(table->size(guard.stream), 0);

  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  d_keys.memset(0, guard.stream);
  d_scores.memset(0, guard.stream);
  d_vectors.memset(0, guard.stream);
  export_and_verify_all(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_scores.get(), guard.stream);
}

void test_basic_when_full(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t INIT_CAPACITY = 1 * 1024 * 1024UL;
  constexpr uint64_t KEY_NUM = INIT_CAPACITY;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(), nullptr,
                                   KEY_NUM);

  StreamGuard guard;
  guard.create();
  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_found.alloc(KEY_NUM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.memset(1, guard.stream);

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);
  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  const auto total_size_after_insert = table->size(guard.stream);
  table->erase(KEY_NUM, d_keys.get(), guard.stream);
  ASSERT_EQ(table->size(guard.stream), 0);
  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), total_size_after_insert);
}

void test_erase_if_pred(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  auto table = std::make_unique<Table>();
  table->init(options);
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys.data(), h_scores.data(), h_vectors.data(), KEY_NUM, INIT_CAPACITY);

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_found.alloc(KEY_NUM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.copy_from_host(h_vectors, guard.stream);

  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);

  K pattern = 100;
  S threshold = 0;
  const size_t erase_num =
      table->template erase_if<EraseIfPredFunctor>(pattern, threshold,
                                                   guard.stream);
  const size_t total_size = table->size(guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(erase_num + total_size, BUCKET_MAX_SIZE);

  d_vectors.memset(0, guard.stream);
  d_found.memset(0, guard.stream);
  find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_found.get(), d_scores.get(), guard.stream);
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  d_scores.copy_to_host(&h_scores, KEY_NUM, guard.stream);
  d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);
  verify_found_values_and_scores(h_keys, h_scores, h_vectors, h_found,
                             BUCKET_MAX_SIZE - erase_num);
}

void test_rehash(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = 4 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = BUCKET_MAX_SIZE * 2;
  constexpr uint64_t TEST_TIMES = 100;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options = make_options(INIT_CAPACITY, MAX_CAPACITY, max_hbm_for_vectors,
                             key_start, BUCKET_MAX_SIZE);
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_found.alloc(KEY_NUM);

  for (size_t round = 0; round < TEST_TIMES; ++round) {
    auto table = std::make_unique<Table>();
    table->init(options);
    create_keys_in_one_buckets<K, S, V, DIM>(
        h_keys.data(), h_scores.data(), h_vectors.data(), KEY_NUM,
        INIT_CAPACITY, BUCKET_MAX_SIZE);
    d_keys.copy_from_host(h_keys, guard.stream);
    d_scores.copy_from_host(h_scores, guard.stream);
    d_vectors.copy_from_host(h_vectors, guard.stream);
    ASSERT_EQ(table->size(guard.stream), 0);

    find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                            d_vectors.get(), options.dim, guard.stream);
    ASSERT_EQ(table->size(guard.stream), KEY_NUM);
    const size_t exported =
        table->export_batch(table->capacity(), 0, d_keys.get(), d_vectors.get(),
                            d_scores.get(), guard.stream);
    ASSERT_EQ(exported, KEY_NUM);
    table->reserve(MAX_CAPACITY, guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    ASSERT_EQ(table->capacity(), MAX_CAPACITY);
    ASSERT_EQ(table->size(guard.stream), KEY_NUM);

    d_vectors.memset(0, guard.stream);
    d_found.memset(0, guard.stream);
    find_into_contiguous(table.get(), BUCKET_MAX_SIZE, d_keys.get(),
                       d_vectors.get(), d_found.get(), d_scores.get(),
                       guard.stream);
    copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
    d_keys.copy_to_host(&h_keys, KEY_NUM, guard.stream);
    d_scores.copy_to_host(&h_scores, KEY_NUM, guard.stream);
    d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);
    verify_found_values_and_scores(h_keys, h_scores, h_vectors, h_found,
                               BUCKET_MAX_SIZE);
    table->clear(guard.stream);
    ASSERT_EQ(table->size(guard.stream), 0);
  }
}

void test_rehash_on_big_batch(size_t max_hbm_for_vectors, int key_start = 0) {
  (void)key_start;
  constexpr uint64_t INIT_CAPACITY = 1024UL;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024UL;
  constexpr uint64_t INIT_KEY_NUM = 1024UL;
  constexpr uint64_t KEY_NUM = 2048UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, MAX_CAPACITY, max_hbm_for_vectors);
  options.max_load_factor = 0.6f;
  StreamGuard guard;
  guard.create();
  auto table = std::make_unique<Table>();
  table->init(options);

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(),
                                   h_vectors.data(), KEY_NUM);

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_found.alloc(KEY_NUM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.copy_from_host(h_vectors, guard.stream);

  find_or_insert_safe(table.get(), INIT_KEY_NUM, d_keys.get(),
                          d_scores.get(), d_vectors.get(), options.dim,
                          guard.stream);
  ASSERT_EQ(table->size(guard.stream), INIT_KEY_NUM);
  ASSERT_EQ(table->capacity(), INIT_CAPACITY * 2);

  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  ASSERT_EQ(table->capacity(), KEY_NUM * 4);
  export_and_verify_all(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_scores.get(), guard.stream);

  d_vectors.memset(0, guard.stream);
  d_scores.memset(0, guard.stream);
  d_found.memset(0, guard.stream);
  find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_found.get(), d_scores.get(), guard.stream);
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  d_keys.copy_to_host(&h_keys, KEY_NUM, guard.stream);
  d_scores.copy_to_host(&h_scores, KEY_NUM, guard.stream);
  d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);
  verify_found_values_and_scores(h_keys, h_scores, h_vectors, h_found, KEY_NUM);

  table->clear(guard.stream);
  ASSERT_EQ(table->size(guard.stream), 0);
}

void test_export_batch_if(size_t max_hbm_for_vectors, int key_start = 0) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t KEY_NUM = 128UL;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  StreamGuard guard;
  guard.create();
  auto table = std::make_unique<Table>();
  table->init(options);

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(),
                                   h_vectors.data(), KEY_NUM);

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<bool> d_found;
  DeviceArray<size_t> d_dump_counter;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_found.alloc(KEY_NUM);
  d_dump_counter.alloc(1);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.copy_from_host(h_vectors, guard.stream);

  const S threshold = npu::hkv::host_nano<S>(guard.stream);
  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), nullptr,
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  d_vectors.memset(0, guard.stream);
  d_found.memset(0, guard.stream);
  find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_found.get(), nullptr, guard.stream);
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);
  verify_found_values_and_scores(h_keys, h_scores, h_vectors, h_found, KEY_NUM,
                             false);

  K pattern = 100;
  d_dump_counter.memset(0, guard.stream);
  table->template export_batch_if<ExportIfPredFunctor>(
      pattern, threshold, table->capacity(), 0, d_dump_counter.get(),
      d_keys.get(), d_vectors.get(), d_scores.get(), guard.stream);
  std::vector<size_t> h_count;
  d_dump_counter.copy_to_host(&h_count, 1, guard.stream);
  d_scores.copy_to_host(&h_scores, KEY_NUM, guard.stream);
  size_t expected_export_count = 0;
  for (size_t i = 0; i < KEY_NUM; ++i) {
    if (h_scores[i] > threshold) {
      ++expected_export_count;
    }
  }
  ASSERT_EQ(h_count[0], expected_export_count);

  d_dump_counter.memset(0, guard.stream);
  table->template export_batch_if<ExportIfPredFunctor>(
      pattern, npu::hkv::host_nano<S>(guard.stream), table->capacity(), 0,
      d_dump_counter.get(), d_keys.get(), d_vectors.get(), d_scores.get(),
      guard.stream);
  d_dump_counter.copy_to_host(&h_count, 1, guard.stream);
  ASSERT_EQ(h_count[0], 0);

  table->clear(guard.stream);
  ASSERT_EQ(table->size(guard.stream), 0);
}

void test_evict_strategy_lru_basic(size_t max_hbm_for_vectors,
                                   int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = BASE_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  StreamGuard guard;
  guard.create();
  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
      0x3FFFFFFFFFFFFFFFUL);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
      0xFFFFFFFFFFFFFFFDUL);

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  for (size_t j = 0; j < DIM; ++j) {
    h_vectors_test[2 * DIM + j] = h_vectors_base[72 * DIM + j];
    h_vectors_test[3 * DIM + j] = h_vectors_base[73 * DIM + j];
  }

  DeviceArray<K> d_keys_temp;
  DeviceArray<S> d_scores_temp;
  DeviceArray<V> d_vectors_temp;
  d_keys_temp.alloc(TEMP_KEY_NUM);
  d_scores_temp.alloc(TEMP_KEY_NUM);
  d_vectors_temp.alloc(TEMP_KEY_NUM * DIM);

  for (size_t round = 0; round < TEST_TIMES; ++round) {
    auto table = std::make_unique<Table>();
    table->init(options);
    ASSERT_EQ(table->size(guard.stream), 0);

    d_keys_temp.copy_from_host(h_keys_base, guard.stream);
    d_scores_temp.copy_from_host(h_scores_base, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_base, guard.stream);
    const S start_ts = npu::hkv::host_nano<S>(guard.stream);
    find_or_insert_safe(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
                            nullptr, d_vectors_temp.get(), options.dim,
                            guard.stream);
    const S end_ts = npu::hkv::host_nano<S>(guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);

    auto h_scores_temp_sorted = h_scores_temp;
    std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());
    ASSERT_GE(h_scores_temp_sorted[0], start_ts);
    ASSERT_LE(h_scores_temp_sorted[BASE_KEY_NUM - 1], end_ts);
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);

    d_keys_temp.copy_from_host(h_keys_test, guard.stream);
    d_scores_temp.copy_from_host(h_scores_test, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_test, guard.stream);
    const S second_start_ts = npu::hkv::host_nano<S>(guard.stream);
    table->assign(TEST_KEY_NUM, d_keys_temp.get(), d_vectors_temp.get(),
                  nullptr, guard.stream);
    find_or_insert_safe(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
                            nullptr, d_vectors_temp.get(), options.dim,
                            guard.stream);
    const S second_end_ts = npu::hkv::host_nano<S>(guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);

    std::vector<S> updated_scores(TEST_KEY_NUM);
    size_t ctr = 0;
    for (size_t i = 0; i < TEMP_KEY_NUM; ++i) {
      if (std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]) !=
          h_keys_test.end()) {
        ASSERT_GT(h_scores_temp[i], BUCKET_MAX_SIZE);
        updated_scores[ctr++] = h_scores_temp[i];
      } else {
        ASSERT_LE(h_scores_temp[i], second_start_ts);
      }
    }
    std::sort(updated_scores.begin(), updated_scores.begin() + ctr);
    ASSERT_GE(updated_scores[0], second_start_ts);
    ASSERT_LE(updated_scores[ctr - 1], second_end_ts);
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);
  }
}

void test_evict_strategy_lfu_basic(size_t max_hbm_for_vectors,
                                   int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = BASE_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr int FREQ_RANGE = 1000;
  using Table = HashTable<K, V, S, EvictStrategy::kLfu>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  StreamGuard guard;
  guard.create();
  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
      0x3FFFFFFFFFFFFFFFUL, FREQ_RANGE);
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
      0xFFFFFFFFFFFFFFFDUL, FREQ_RANGE);
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  h_scores_test[2] = h_keys_base[72] % FREQ_RANGE;
  h_scores_test[3] = h_keys_base[73] % FREQ_RANGE;
  for (size_t j = 0; j < DIM; ++j) {
    h_vectors_test[2 * DIM + j] = h_vectors_base[72 * DIM + j];
    h_vectors_test[3 * DIM + j] = h_vectors_base[73 * DIM + j];
  }

  DeviceArray<K> d_keys_temp;
  DeviceArray<S> d_scores_temp;
  DeviceArray<V> d_vectors_temp;
  d_keys_temp.alloc(TEMP_KEY_NUM);
  d_scores_temp.alloc(TEMP_KEY_NUM);
  d_vectors_temp.alloc(TEMP_KEY_NUM * DIM);

  S global_epoch = 1;
  for (size_t round = 0; round < TEST_TIMES; ++round) {
    auto table = std::make_unique<Table>();
    table->init(options);
    ASSERT_EQ(table->size(guard.stream), 0);

    d_keys_temp.copy_from_host(h_keys_base, guard.stream);
    d_scores_temp.copy_from_host(h_scores_base, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_base, guard.stream);
    find_or_insert_safe(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
                            d_scores_temp.get(), d_vectors_temp.get(),
                            options.dim, guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);
    for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
      ASSERT_EQ(h_scores_temp[i], h_keys_temp[i] % FREQ_RANGE);
    }
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);

    ++global_epoch;
    d_keys_temp.copy_from_host(h_keys_test, guard.stream);
    d_scores_temp.copy_from_host(h_scores_test, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_test, guard.stream);
    table->assign(TEST_KEY_NUM, d_keys_temp.get(), d_vectors_temp.get(),
                  d_scores_temp.get(), guard.stream);
    find_or_insert_safe(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
                            d_scores_temp.get(), d_vectors_temp.get(),
                            options.dim, guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);
    for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
      const bool in_base =
          std::find(h_keys_base.begin(), h_keys_base.end(), h_keys_temp[i]) !=
          h_keys_base.end();
      const bool in_test =
          std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]) !=
          h_keys_test.end();
      if (in_base && in_test) {
        ASSERT_EQ(h_scores_temp[i], (h_keys_temp[i] % FREQ_RANGE) * 3);
      } else {
        ASSERT_EQ(h_scores_temp[i], h_keys_temp[i] % FREQ_RANGE);
      }
    }
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);
  }
}

void test_evict_strategy_epochlru_basic(size_t max_hbm_for_vectors,
                                        int key_start = 0) {
  constexpr int RSHIFT_ON_NANO = 20;
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = BASE_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  using Table = HashTable<K, V, S, EvictStrategy::kEpochLru>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  StreamGuard guard;
  guard.create();
  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
      0x3FFFFFFFFFFFFFFFUL);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
      0xFFFFFFFFFFFFFFFDUL);
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  for (size_t j = 0; j < DIM; ++j) {
    h_vectors_test[2 * DIM + j] = h_vectors_base[72 * DIM + j];
    h_vectors_test[3 * DIM + j] = h_vectors_base[73 * DIM + j];
  }

  DeviceArray<K> d_keys_temp;
  DeviceArray<S> d_scores_temp;
  DeviceArray<V> d_vectors_temp;
  d_keys_temp.alloc(TEMP_KEY_NUM);
  d_scores_temp.alloc(TEMP_KEY_NUM);
  d_vectors_temp.alloc(TEMP_KEY_NUM * DIM);

  S global_epoch = 1;
  for (size_t round = 0; round < TEST_TIMES; ++round) {
    auto table = std::make_unique<Table>();
    table->init(options);
    ASSERT_EQ(table->size(guard.stream), 0);

    d_keys_temp.copy_from_host(h_keys_base, guard.stream);
    d_scores_temp.copy_from_host(h_scores_base, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_base, guard.stream);
    const S start_ts =
        (npu::hkv::host_nano<S>(guard.stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    table->set_global_epoch(global_epoch);
    find_or_insert_safe(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
                            nullptr, d_vectors_temp.get(), options.dim,
                            guard.stream);
    const S end_ts =
        (npu::hkv::host_nano<S>(guard.stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);
    auto h_scores_temp_sorted = h_scores_temp;
    std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());
    ASSERT_GE(h_scores_temp_sorted[0], ((global_epoch << 32) | start_ts));
    ASSERT_LE(h_scores_temp_sorted[BASE_KEY_NUM - 1],
              ((global_epoch << 32) | end_ts));
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);

    ++global_epoch;
    d_keys_temp.copy_from_host(h_keys_test, guard.stream);
    d_scores_temp.copy_from_host(h_scores_test, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_test, guard.stream);
    const S second_start_ts =
        (npu::hkv::host_nano<S>(guard.stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    table->set_global_epoch(global_epoch);
    table->assign(TEST_KEY_NUM, d_keys_temp.get(), d_vectors_temp.get(),
                  nullptr, guard.stream);
    find_or_insert_safe(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
                            nullptr, d_vectors_temp.get(), options.dim,
                            guard.stream);
    const S second_end_ts =
        (npu::hkv::host_nano<S>(guard.stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);
    std::vector<S> updated_scores(TEST_KEY_NUM);
    size_t ctr = 0;
    for (size_t i = 0; i < TEMP_KEY_NUM; ++i) {
      if (std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]) !=
          h_keys_test.end()) {
        ASSERT_GE(h_scores_temp[i], ((global_epoch << 32) | second_start_ts));
        updated_scores[ctr++] = h_scores_temp[i];
      } else {
        ASSERT_LE(h_scores_temp[i], ((global_epoch << 32) | second_start_ts));
      }
    }
    std::sort(updated_scores.begin(), updated_scores.begin() + ctr);
    ASSERT_GE(updated_scores[0], ((global_epoch << 32) | second_start_ts));
    ASSERT_LE(updated_scores[ctr - 1],
              ((global_epoch << 32) | second_end_ts));
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);
  }
}

void test_evict_strategy_epochlfu_basic(size_t max_hbm_for_vectors,
                                        int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = BASE_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr int FREQ_RANGE = 1000;
  using Table = HashTable<K, V, S, EvictStrategy::kEpochLfu>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  StreamGuard guard;
  guard.create();
  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
      0x3FFFFFFFFFFFFFFFUL, FREQ_RANGE);
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
      0xFFFFFFFFFFFFFFFDUL, FREQ_RANGE);
  // Simulate overflow of low 32bits.
  h_scores_base[71] = static_cast<S>(std::numeric_limits<uint32_t>::max() -
                                     static_cast<uint32_t>(1));
  h_keys_test[1] = h_keys_base[71];
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  h_scores_test[1] = h_scores_base[71];
  h_scores_test[2] = h_keys_base[72] % FREQ_RANGE;
  h_scores_test[3] = h_keys_base[73] % FREQ_RANGE;
  for (size_t j = 0; j < DIM; ++j) {
    h_vectors_test[1 * DIM + j] = h_vectors_base[71 * DIM + j];
    h_vectors_test[2 * DIM + j] = h_vectors_base[72 * DIM + j];
    h_vectors_test[3 * DIM + j] = h_vectors_base[73 * DIM + j];
  }

  DeviceArray<K> d_keys_temp;
  DeviceArray<S> d_scores_temp;
  DeviceArray<V> d_vectors_temp;
  d_keys_temp.alloc(TEMP_KEY_NUM);
  d_scores_temp.alloc(TEMP_KEY_NUM);
  d_vectors_temp.alloc(TEMP_KEY_NUM * DIM);

  S global_epoch = 1;
  for (size_t round = 0; round < TEST_TIMES; ++round) {
    auto table = std::make_unique<Table>();
    table->init(options);
    ASSERT_EQ(table->size(guard.stream), 0);

    d_keys_temp.copy_from_host(h_keys_base, guard.stream);
    d_scores_temp.copy_from_host(h_scores_base, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_base, guard.stream);
    table->set_global_epoch(global_epoch);
    find_or_insert_safe(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
                            d_scores_temp.get(), d_vectors_temp.get(),
                            options.dim, guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);
    for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
      const S original = h_keys_temp[i] == h_keys_base[71]
                             ? h_scores_base[71]
                             : (h_keys_temp[i] % FREQ_RANGE);
      ASSERT_EQ(h_scores_temp[i],
                make_expected_score_for_epochlfu<S>(global_epoch, original));
    }
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);

    ++global_epoch;
    d_keys_temp.copy_from_host(h_keys_test, guard.stream);
    d_scores_temp.copy_from_host(h_scores_test, guard.stream);
    d_vectors_temp.copy_from_host(h_vectors_test, guard.stream);
    table->set_global_epoch(global_epoch);
    table->assign(TEST_KEY_NUM, d_keys_temp.get(), d_vectors_temp.get(),
                  d_scores_temp.get(), guard.stream);
    find_or_insert_safe(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
                            d_scores_temp.get(), d_vectors_temp.get(),
                            options.dim, guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys_temp.get(),
                 d_vectors_temp.get(), d_scores_temp.get(), &h_keys_temp,
                 &h_vectors_temp, &h_scores_temp, guard.stream);
    ASSERT_TRUE(std::find(h_keys_temp.begin(), h_keys_temp.end(),
                          h_keys_base[71]) != h_keys_temp.end());
    for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
      const bool in_base =
          std::find(h_keys_base.begin(), h_keys_base.end(), h_keys_temp[i]) !=
          h_keys_base.end();
      const bool in_test =
          std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]) !=
          h_keys_test.end();
      S expected_score = 0;
      if (in_base && in_test) {
        if (h_keys_temp[i] == h_keys_base[71]) {
          expected_score =
              make_expected_score_for_epochlfu<S>(global_epoch,
                                                  h_scores_base[71] * 2);
        } else {
          expected_score = make_expected_score_for_epochlfu<S>(
              global_epoch, (h_keys_temp[i] % FREQ_RANGE) * 3);
        }
      } else {
        const S score_epoch = global_epoch - static_cast<S>(in_base);
        const S original = h_keys_temp[i] == h_keys_base[71]
                               ? h_scores_base[71]
                               : (h_keys_temp[i] % FREQ_RANGE);
        expected_score = make_expected_score_for_epochlfu<S>(score_epoch,
                                                             original);
      }
      ASSERT_EQ(h_scores_temp[i], expected_score);
    }
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);
  }
}

void test_evict_strategy_customized_basic(size_t max_hbm_for_vectors,
                                          int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 128;
  constexpr uint64_t TEMP_KEY_NUM = BASE_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  StreamGuard guard;
  guard.create();
  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
      0x3FFFFFFFFFFFFFFFUL);
  constexpr S BASE_SCORE_START = 1000;
  for (size_t i = 0; i < BASE_KEY_NUM; ++i) {
    h_scores_base[i] = BASE_SCORE_START + i;
  }
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
      0xFFFFFFFFFFFFFFFDUL);
  constexpr S TEST_SCORE_START = BASE_SCORE_START + BASE_KEY_NUM;
  for (size_t i = 0; i < TEST_KEY_NUM; ++i) {
    h_scores_test[i] = TEST_SCORE_START + i;
  }
  for (size_t i = 64; i < TEST_KEY_NUM; ++i) {
    h_keys_test[i] = h_keys_base[i];
    for (size_t j = 0; j < DIM; ++j) {
      h_vectors_test[i * DIM + j] = h_vectors_base[i * DIM + j];
    }
  }

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  d_keys.alloc(TEMP_KEY_NUM);
  d_scores.alloc(TEMP_KEY_NUM);
  d_vectors.alloc(TEMP_KEY_NUM * DIM);

  for (size_t round = 0; round < TEST_TIMES; ++round) {
    auto table = std::make_unique<Table>();
    table->init(options);
    ASSERT_EQ(table->size(guard.stream), 0);

    d_keys.copy_from_host(h_keys_base, guard.stream);
    d_scores.copy_from_host(h_scores_base, guard.stream);
    d_vectors.copy_from_host(h_vectors_base, guard.stream);
    find_or_insert_safe(table.get(), BASE_KEY_NUM, d_keys.get(),
                            d_scores.get(), d_vectors.get(), options.dim,
                            guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream);
    auto h_scores_temp_sorted = h_scores_temp;
    std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());
    auto expected_range = range<S, TEMP_KEY_NUM>(BASE_SCORE_START);
    ASSERT_TRUE(std::equal(h_scores_temp_sorted.begin(),
                           h_scores_temp_sorted.end(),
                           expected_range.begin()));
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);

    d_keys.copy_from_host(h_keys_test, guard.stream);
    d_scores.copy_from_host(h_scores_test, guard.stream);
    d_vectors.copy_from_host(h_vectors_test, guard.stream);
    table->assign(TEST_KEY_NUM, d_keys.get(), d_vectors.get(), d_scores.get(),
                  guard.stream);
    find_or_insert_safe(table.get(), TEST_KEY_NUM, d_keys.get(),
                            d_scores.get(), d_vectors.get(), options.dim,
                            guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream);
    h_scores_temp_sorted = h_scores_temp;
    std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());
    auto expected_range_test = range<S, TEST_KEY_NUM>(TEST_SCORE_START);
    ASSERT_TRUE(std::equal(h_scores_temp_sorted.begin(),
                           h_scores_temp_sorted.end(),
                           expected_range_test.begin()));
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);
  }
}

void test_evict_strategy_customized_advanced(size_t max_hbm_for_vectors,
                                             int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 8;
  constexpr uint64_t TEMP_KEY_NUM = BASE_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 256;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, INIT_CAPACITY, max_hbm_for_vectors, key_start);
  StreamGuard guard;
  guard.create();
  std::vector<K> h_keys_base(BASE_KEY_NUM);
  std::vector<S> h_scores_base(BASE_KEY_NUM);
  std::vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  std::vector<K> h_keys_test(TEST_KEY_NUM);
  std::vector<S> h_scores_test(TEST_KEY_NUM);
  std::vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  std::vector<K> h_keys_temp(TEMP_KEY_NUM);
  std::vector<S> h_scores_temp(TEMP_KEY_NUM);
  std::vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
      0x3FFFFFFFFFFFFFFFUL);
  constexpr S BASE_SCORE_START = 1000;
  for (size_t i = 0; i < BASE_KEY_NUM; ++i) {
    h_scores_base[i] = BASE_SCORE_START + i;
  }
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
      0xFFFFFFFFFFFFFFFDUL);
  h_keys_test[4] = h_keys_base[72];
  h_keys_test[5] = h_keys_base[73];
  h_keys_test[6] = h_keys_base[74];
  h_keys_test[7] = h_keys_base[75];

  // replace four new keys to lower scores, would not be inserted.
  h_scores_test[0] = 20;
  h_scores_test[1] = 78;
  h_scores_test[2] = 97;
  h_scores_test[3] = 98;

  // replace three exist keys to new scores, just refresh the score for them.
  h_scores_test[4] = 99;
  h_scores_test[5] = 1010;
  h_scores_test[6] = 1020;
  h_scores_test[7] = 1035;

  for (size_t i = 4; i < TEST_KEY_NUM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      h_vectors_test[i * DIM + j] =
          static_cast<V>(h_keys_test[i] * 0.00001);
    }
  }

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  d_keys.alloc(TEMP_KEY_NUM);
  d_scores.alloc(TEMP_KEY_NUM);
  d_vectors.alloc(TEMP_KEY_NUM * DIM);

  for (size_t round = 0; round < TEST_TIMES; ++round) {
    auto table = std::make_unique<Table>();
    table->init(options);
    ASSERT_EQ(table->size(guard.stream), 0);

    d_keys.copy_from_host(h_keys_base, guard.stream);
    d_scores.copy_from_host(h_scores_base, guard.stream);
    d_vectors.copy_from_host(h_vectors_base, guard.stream);
    find_or_insert_safe(table.get(), BASE_KEY_NUM, d_keys.get(),
                            d_scores.get(), d_vectors.get(), options.dim,
                            guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream);
    auto h_scores_temp_sorted = h_scores_temp;
    std::sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());
    auto expected_range = range<S, TEMP_KEY_NUM>(BASE_SCORE_START);
    ASSERT_TRUE(std::equal(h_scores_temp_sorted.begin(),
                           h_scores_temp_sorted.end(),
                           expected_range.begin()));
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);

    d_keys.copy_from_host(h_keys_test, guard.stream);
    d_scores.copy_from_host(h_scores_test, guard.stream);
    d_vectors.copy_from_host(h_vectors_test, guard.stream);
    table->assign(TEST_KEY_NUM, d_keys.get(), d_vectors.get(), d_scores.get(),
                  guard.stream);
    find_or_insert_safe(table.get(), TEST_KEY_NUM, d_keys.get(),
                            d_scores.get(), d_vectors.get(), options.dim,
                            guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream);
    for (size_t i = 0; i < TEST_KEY_NUM; ++i) {
      if (i < 4) {
        ASSERT_EQ(h_keys_temp.end(),
                  std::find(h_keys_temp.begin(), h_keys_temp.end(),
                            h_keys_test[i]));
      } else {
        ASSERT_NE(h_keys_temp.end(),
                  std::find(h_keys_temp.begin(), h_keys_temp.end(),
                            h_keys_test[i]));
      }
    }
    for (size_t i = 0; i < TEMP_KEY_NUM; ++i) {
      if (h_keys_temp[i] == h_keys_test[4]) {
        ASSERT_EQ(h_scores_temp[i], h_scores_test[4]);
      }
      if (h_keys_temp[i] == h_keys_test[5]) {
        ASSERT_EQ(h_scores_temp[i], h_scores_test[5]);
      }
      if (h_keys_temp[i] == h_keys_test[6]) {
        ASSERT_EQ(h_scores_temp[i], h_scores_test[6]);
      }
      if (h_keys_temp[i] == h_keys_test[7]) {
        ASSERT_EQ(h_scores_temp[i], h_scores_test[7]);
      }
    }
    verify_exported_values(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE);
  }
}

template <typename VType, int Dim>
void test_value_type_hbm_mode() {
  std::cout << "size of V: " << sizeof(VType) << ", dim: " << Dim << std::endl;
  constexpr uint64_t BUCKET_MAX_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_TIMES = 2UL;
  using Table = HashTable<K, VType, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options{};
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = Dim;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = GB(16);
  set_simd_value_move_enabled(false);

  StreamGuard guard;
  guard.create();
  auto table = std::make_unique<Table>();
  table->init(options);

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<VType> h_vectors(KEY_NUM * Dim);

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<VType> d_vectors;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * Dim);

  ASSERT_EQ(table->size(guard.stream), 0);

  K start_key = 1;
  for (size_t round = 0; round < TEST_TIMES; ++round) {
    for (K i = 0; i < KEY_NUM; ++i) {
      h_keys[i] = start_key + static_cast<K>(i);
      h_scores[i] = h_keys[i];
      for (size_t j = 0; j < options.dim; ++j) {
        h_vectors[i * options.dim + j] = static_cast<VType>(h_keys[i] * 0.1);
      }
    }
    start_key += KEY_NUM;

    // Step1 : insert new Keys.
    const uint64_t table_size_before = table->size(guard.stream);
    d_keys.copy_from_host(h_keys, guard.stream);
    d_scores.copy_from_host(h_scores, guard.stream);
    d_vectors.copy_from_host(h_vectors, guard.stream);
    find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                        d_vectors.get(), options.dim, guard.stream);
    uint64_t table_size_after = table->size(guard.stream);
    ASSERT_LE(table_size_after, table_size_before + KEY_NUM);

    // Step2 : find new keys.
    d_vectors.memset(0, guard.stream);
    d_scores.memset(0, guard.stream);
    find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                        d_vectors.get(), options.dim, guard.stream);
    ASSERT_EQ(table->size(guard.stream), table_size_after);
    d_vectors.copy_to_host(&h_vectors, KEY_NUM * Dim, guard.stream);
    size_t found_num = 0;
    size_t value_diff_cnt = 0;
    for (size_t i = 0; i < KEY_NUM; ++i) {
      if (h_vectors[i * options.dim] == static_cast<VType>(h_keys[i] * 0.1)) {
        ++found_num;
      }
      for (size_t j = 0; j < options.dim; ++j) {
        if (h_vectors[i * options.dim + j] !=
            static_cast<VType>(h_keys[i] * 0.1)) {
          ++value_diff_cnt;
          break;
        }
      }
    }
    ASSERT_EQ(found_num, KEY_NUM);
    ASSERT_EQ(value_diff_cnt, 0);
    std::cout << "Check find_or_insert behavior got "
              << " key_miss_cnt: " << KEY_NUM - found_num
              << " value_diff_cnt: " << value_diff_cnt
              << " while table_size_before: " << table_size_before
              << ", while table_size_after: " << table_size_after
              << ", while len: " << KEY_NUM << std::endl;

    // Step3 : update old keys.
    for (size_t i = 0; i < KEY_NUM; ++i) {
      h_scores[i] = h_keys[i];
      for (size_t j = 0; j < options.dim; ++j) {
        h_vectors[i * options.dim + j] = static_cast<VType>(h_keys[i] * 0.2);
      }
    }
    const uint64_t assign_size_before = table->size(guard.stream);
    d_scores.copy_from_host(h_scores, guard.stream);
    d_vectors.copy_from_host(h_vectors, guard.stream);
    table->assign(KEY_NUM, d_keys.get(), d_vectors.get(), d_scores.get(),
                  guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    table_size_after = table->size(guard.stream);
    ASSERT_EQ(assign_size_before, table_size_after);

    // Step4 : find old keys.
    d_vectors.memset(0, guard.stream);
    d_scores.memset(0, guard.stream);
    find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                        d_vectors.get(), options.dim, guard.stream);
    ASSERT_EQ(table->size(guard.stream), table_size_after);
    d_vectors.copy_to_host(&h_vectors, KEY_NUM * Dim, guard.stream);
    found_num = 0;
    value_diff_cnt = 0;
    for (size_t i = 0; i < KEY_NUM; ++i) {
      if (h_vectors[i * options.dim] == static_cast<VType>(h_keys[i] * 0.2)) {
        ++found_num;
      }
      for (size_t j = 0; j < options.dim; ++j) {
        if (h_vectors[i * options.dim + j] !=
            static_cast<VType>(h_keys[i] * 0.2)) {
          ++value_diff_cnt;
          break;
        }
      }
    }
    ASSERT_EQ(found_num, KEY_NUM);
    ASSERT_EQ(value_diff_cnt, 0);
    std::cout << "Check  assign        behavior got "
              << " key_miss_cnt: " << KEY_NUM - found_num
              << " value_diff_cnt: " << value_diff_cnt
              << " while table_size_before: " << assign_size_before
              << ", while table_size_after: " << table_size_after
              << ", while len: " << KEY_NUM << std::endl;
  }
}

template <typename KType, typename VType, typename SType, typename Table,
          size_t dim = 16>
void check_assign_on_epoch_lfu(Table* table,
                           test_util::KVMSBuffer<KType, VType, SType>* data,
                           test_util::KVMSBuffer<KType, VType, SType>* evict,
                           size_t len, aclrtStream stream,
                           unsigned int global_epoch) {
  std::map<KType, test_util::ValueArray<VType, dim>> values_before;
  std::map<KType, test_util::ValueArray<VType, dim>> values_after;
  std::unordered_map<KType, SType> scores_before;
  std::map<KType, SType> scores_after;
  std::map<KType, SType> scores_current_batch;
  std::map<KType, SType> scores_current_evict;

  for (size_t i = 0; i < len; ++i) {
    scores_current_batch[data->keys_ptr(false)[i]] = data->scores_ptr(false)[i];
  }

  const size_t table_size_before = table->size(stream);
  const size_t cap = table_size_before + len;
  DeviceArray<KType> d_tmp_keys;
  DeviceArray<VType> d_tmp_values;
  DeviceArray<SType> d_tmp_scores;
  d_tmp_keys.alloc(cap);
  d_tmp_values.alloc(cap * dim);
  d_tmp_scores.alloc(cap);
  d_tmp_keys.memset(0, stream);
  d_tmp_values.memset(0, stream);
  d_tmp_scores.memset(0, stream);

  std::vector<KType> h_tmp_keys;
  std::vector<VType> h_tmp_values;
  std::vector<SType> h_tmp_scores;

  const size_t table_size_verify0 =
      table->export_batch(table->capacity(), 0, d_tmp_keys.get(),
                          d_tmp_values.get(), d_tmp_scores.get(), stream);
  ASSERT_EQ(table_size_before, table_size_verify0);
  d_tmp_keys.copy_to_host(&h_tmp_keys, table_size_before, stream);
  d_tmp_values.copy_to_host(&h_tmp_values, table_size_before * dim, stream);
  d_tmp_scores.copy_to_host(&h_tmp_scores, table_size_before, stream);

  h_tmp_keys.resize(cap);
  h_tmp_values.resize(cap * dim);
  h_tmp_scores.resize(cap);
  std::copy(data->keys_ptr(false), data->keys_ptr(false) + len,
            h_tmp_keys.begin() + static_cast<ptrdiff_t>(table_size_before));
  std::copy(data->values_ptr(false), data->values_ptr(false) + len * dim,
            h_tmp_values.begin() +
                static_cast<ptrdiff_t>(table_size_before * dim));
  std::copy(data->scores_ptr(false), data->scores_ptr(false) + len,
            h_tmp_scores.begin() + static_cast<ptrdiff_t>(table_size_before));

  for (size_t i = 0; i < cap; ++i) {
    auto* vec = reinterpret_cast<test_util::ValueArray<VType, dim>*>(
        h_tmp_values.data() + i * dim);
    values_before[h_tmp_keys[i]] = *vec;
  }
  for (size_t i = 0; i < table_size_before; ++i) {
    scores_before[h_tmp_keys[i]] = h_tmp_scores[i];
  }

  table->set_global_epoch(global_epoch);
  table->assign(len, data->keys_ptr(), data->values_ptr(), data->scores_ptr(),
                stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  const size_t table_size_verify1 =
      table->export_batch(table->capacity(), 0, d_tmp_keys.get(),
                          d_tmp_values.get(), d_tmp_scores.get(), stream);
  ASSERT_EQ(table_size_verify1, table_size_before);
  d_tmp_keys.copy_to_host(&h_tmp_keys, table_size_before, stream);
  d_tmp_values.copy_to_host(&h_tmp_values, table_size_before * dim, stream);
  d_tmp_scores.copy_to_host(&h_tmp_scores, table_size_before, stream);

  size_t assign_score_error_cnt = 0;
  for (int64_t i = static_cast<int64_t>(table_size_before) - 1; i >= 0; --i) {
    auto* vec = reinterpret_cast<test_util::ValueArray<VType, dim>*>(
        h_tmp_values.data() + static_cast<size_t>(i) * dim);
    values_after[h_tmp_keys[static_cast<size_t>(i)]] = *vec;
    scores_after[h_tmp_keys[static_cast<size_t>(i)]] =
        h_tmp_scores[static_cast<size_t>(i)];
  }
  for (auto it : scores_current_batch) {
    const KType key = it.first;
    const KType score = it.second;
    const SType current_score = scores_after[key];
    if (scores_before.find(key) != scores_before.end()) {
      const SType score_before_insert = scores_before[key];
      const bool valid =
          ((current_score >> 32) == global_epoch) &&
          ((current_score & 0xFFFFFFFF) ==
           ((0xFFFFFFFF & score_before_insert) + (0xFFFFFFFF & score)));
      if (!valid) {
        ++assign_score_error_cnt;
      }
    }
  }
  ASSERT_EQ(assign_score_error_cnt, 0);

  for (int64_t i = 0; i < static_cast<int64_t>(table_size_before); ++i) {
    values_before[h_tmp_keys[static_cast<size_t>(i)]] =
        values_after[h_tmp_keys[static_cast<size_t>(i)]];
    scores_before[h_tmp_keys[static_cast<size_t>(i)]] =
        scores_after[h_tmp_keys[static_cast<size_t>(i)]];
  }
  values_after.clear();
  scores_after.clear();

  table->set_global_epoch(global_epoch);
  const size_t filtered_len = table->insert_and_evict(
      len, data->keys_ptr(), data->values_ptr(), data->scores_ptr(),
      evict->keys_ptr(), evict->values_ptr(), evict->scores_ptr(), stream);
  evict->sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  for (size_t i = 0; i < filtered_len; ++i) {
    scores_current_evict[evict->keys_ptr(false)[i]] = evict->scores_ptr(false)[i];
  }

  const size_t table_size_after = table->size(stream);
  const size_t table_size_verify2 =
      table->export_batch(table->capacity(), 0, d_tmp_keys.get(),
                          d_tmp_values.get(), d_tmp_scores.get(), stream);
  ASSERT_EQ(table_size_verify2, table_size_after);
  d_tmp_keys.copy_to_host(&h_tmp_keys, table_size_after, stream);
  d_tmp_values.copy_to_host(&h_tmp_values, table_size_after * dim, stream);
  d_tmp_scores.copy_to_host(&h_tmp_scores, table_size_after, stream);
  h_tmp_keys.resize(table_size_after + filtered_len);
  h_tmp_values.resize((table_size_after + filtered_len) * dim);
  h_tmp_scores.resize(table_size_after + filtered_len);
  std::copy(evict->keys_ptr(false), evict->keys_ptr(false) + filtered_len,
            h_tmp_keys.begin() + static_cast<ptrdiff_t>(table_size_after));
  std::copy(evict->values_ptr(false),
            evict->values_ptr(false) + filtered_len * dim,
            h_tmp_values.begin() +
                static_cast<ptrdiff_t>(table_size_after * dim));
  std::copy(evict->scores_ptr(false), evict->scores_ptr(false) + filtered_len,
            h_tmp_scores.begin() + static_cast<ptrdiff_t>(table_size_after));

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt1 = 0;
  size_t score_error_cnt2 = 0;
  const size_t new_cap = table_size_after + filtered_len;
  for (int64_t i = static_cast<int64_t>(new_cap) - 1; i >= 0; --i) {
    const size_t idx = static_cast<size_t>(i);
    auto* vec = reinterpret_cast<test_util::ValueArray<VType, dim>*>(
        h_tmp_values.data() + idx * dim);
    values_after[h_tmp_keys[idx]] = *vec;
    scores_after[h_tmp_keys[idx]] = h_tmp_scores[idx];
    if (idx >= (new_cap - filtered_len) &&
        !((h_tmp_scores[idx] >> 32) < (global_epoch - 2))) {
      ++score_error_cnt1;
    }
  }

  for (auto it : scores_current_batch) {
    const KType key = it.first;
    const KType score = it.second;
    const SType current_score = scores_after[key];
    SType score_before_insert = 0;
    if (values_after.find(key) != values_after.end() &&
        scores_current_evict.find(key) == scores_current_evict.end()) {
      score_before_insert = scores_before[key];
    }
    const bool valid =
        ((current_score >> 32) == global_epoch) &&
        ((current_score & 0xFFFFFFFF) ==
         ((0xFFFFFFFF & score_before_insert) + (0xFFFFFFFF & score)));
    if (!valid) {
      ++score_error_cnt2;
    }
  }

  for (auto& it : values_before) {
    if (values_after.find(it.first) == values_after.end()) {
      ++key_miss_cnt;
      continue;
    }
    test_util::ValueArray<VType, dim>& vec0 = it.second;
    test_util::ValueArray<VType, dim>& vec1 = values_after.at(it.first);
    for (size_t j = 0; j < dim; ++j) {
      if (vec0[j] != vec1[j]) {
        ++value_diff_cnt;
        break;
      }
    }
  }

  std::cout << "Check insert_and_evict behavior got "
            << "key_miss_cnt: " << key_miss_cnt
            << ", value_diff_cnt: " << value_diff_cnt
            << ", score_error_cnt1: " << score_error_cnt1
            << ", score_error_cnt2: " << score_error_cnt2
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << std::endl;
  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
  ASSERT_EQ(score_error_cnt1, 0);
  ASSERT_EQ(score_error_cnt2, 0);
}

void test_assign_advanced_on_epochlfu(size_t max_hbm_for_vectors) {
  const size_t U = 1024 * 1024;
  const size_t B = 100000;
  constexpr size_t dim = 16;
  using Table = HashTable<K, V, S, EvictStrategy::kEpochLfu>;

  init_env();
  TableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = U;
  opt.max_hbm_for_vectors = GB(max_hbm_for_vectors);
  set_simd_value_move_enabled(max_hbm_for_vectors == 0);
  opt.max_bucket_size = 128;
  opt.dim = dim;
  StreamGuard guard;
  guard.create();
  auto table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<K, V, S> evict_buffer;
  test_util::KVMSBuffer<K, V, S> data_buffer;
  test_util::KVMSBuffer<K, V, S> pre_data_buffer;
  evict_buffer.reserve(B, dim, guard.stream);
  evict_buffer.to_zeros(guard.stream);
  data_buffer.reserve(B, dim, guard.stream);
  pre_data_buffer.reserve(B, dim, guard.stream);

  int freq_range = 100;
  float repeat_rate = 0.9f;
  for (unsigned int global_epoch = 1; global_epoch <= 20; ++global_epoch) {
    repeat_rate = global_epoch <= 1 ? 0.0f : 0.1f;
    if (global_epoch <= 1) {
      create_random_keys_advanced<K, S, V>(
          dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
          data_buffer.values_ptr(false), static_cast<int>(B), B * 32,
          freq_range);
    } else {
      create_random_keys_advanced<K, S, V>(
          dim, data_buffer.keys_ptr(false), pre_data_buffer.keys_ptr(false),
          data_buffer.scores_ptr(false), data_buffer.values_ptr(false),
          static_cast<int>(B), B * 32, freq_range, repeat_rate);
    }
    data_buffer.sync_data(true, guard.stream);
    if (global_epoch <= 1) {
      pre_data_buffer.copy_from(data_buffer, guard.stream);
    }
    check_assign_on_epoch_lfu<K, V, S, Table, dim>(
        table.get(), &data_buffer, &evict_buffer, B, guard.stream,
        global_epoch);
    pre_data_buffer.copy_from(data_buffer, guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  }
}

void test_evict_strategy_customized_correct_rate(size_t max_hbm_for_vectors,
                                                 int key_start = 0) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024UL;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr float EXPECTED_CORRECT_RATE = 0.964f;
  constexpr int ROUNDS = 12;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, MAX_CAPACITY, max_hbm_for_vectors, key_start,
                  MAX_BUCKET_SIZE);
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys_base(BATCH_SIZE);
  std::vector<S> h_scores_base(BATCH_SIZE);
  std::vector<V> h_vectors_base(BATCH_SIZE * DIM);
  std::vector<K> h_keys_temp(MAX_CAPACITY);
  std::vector<S> h_scores_temp(MAX_CAPACITY);
  std::vector<V> h_vectors_temp(MAX_CAPACITY * DIM);

  DeviceArray<K> d_keys_temp;
  DeviceArray<S> d_scores_temp;
  DeviceArray<V> d_vectors_temp;
  d_keys_temp.alloc(MAX_CAPACITY);
  d_scores_temp.alloc(MAX_CAPACITY);
  d_vectors_temp.alloc(MAX_CAPACITY * DIM);

  size_t global_start_key = 100000;
  for (size_t test_time = 0; test_time < TEST_TIMES; ++test_time) {
    auto table = std::make_unique<Table>();
    table->init(options);
    size_t start_key = global_start_key;
    ASSERT_EQ(table->size(guard.stream), 0);

    for (int round = 0; round < ROUNDS; ++round) {
      const size_t expected_min_key = global_start_key + INIT_CAPACITY * round;
      const size_t expected_max_key =
          global_start_key + INIT_CAPACITY * (round + 1) - 1;
      const size_t expected_table_size =
          round == 0 ? static_cast<size_t>(EXPECTED_CORRECT_RATE *
                                           INIT_CAPACITY)
                     : INIT_CAPACITY;

      for (size_t step = 0; step < STEPS; ++step) {
        create_continuous_keys<K, S, V, DIM>(
            h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
            BATCH_SIZE, start_key);
        start_key += BATCH_SIZE;
        d_keys_temp.copy_from_host(h_keys_base, guard.stream);
        d_scores_temp.copy_from_host(h_scores_base, guard.stream);
        d_vectors_temp.copy_from_host(h_vectors_base, guard.stream);
        table->assign(BATCH_SIZE, d_keys_temp.get(), d_vectors_temp.get(),
                      d_scores_temp.get(), guard.stream);
        find_or_insert_safe(table.get(), BATCH_SIZE, d_keys_temp.get(),
                                d_scores_temp.get(), d_vectors_temp.get(),
                                options.dim, guard.stream);
      }

      const size_t total_size = table->size(guard.stream);
      ACL_CHECK(aclrtSynchronizeStream(guard.stream));
      ASSERT_GE(total_size, expected_table_size);
      ASSERT_EQ(MAX_CAPACITY, table->capacity());

      const size_t dump_counter =
          table->export_batch(MAX_CAPACITY, 0, d_keys_temp.get(),
                              d_vectors_temp.get(), d_scores_temp.get(),
                              guard.stream);
      ACL_CHECK(aclrtSynchronizeStream(guard.stream));
      d_keys_temp.copy_to_host(&h_keys_temp, MAX_CAPACITY, guard.stream);
      d_scores_temp.copy_to_host(&h_scores_temp, MAX_CAPACITY, guard.stream);
      d_vectors_temp.copy_to_host(&h_vectors_temp, MAX_CAPACITY * DIM,
                                guard.stream);

      size_t bigger_score_counter = 0;
      K max_key = 0;
      size_t values_error_counter = 0;
      for (size_t i = 0; i < dump_counter; ++i) {
        ASSERT_EQ(h_keys_temp[i], h_scores_temp[i]);
        max_key = std::max(max_key, h_keys_temp[i]);
        if (h_scores_temp[i] >= expected_min_key) {
          ++bigger_score_counter;
        }
        for (size_t j = 0; j < DIM; ++j) {
          if (h_vectors_temp[i * DIM + j] !=
              static_cast<float>(h_keys_temp[i] * 0.00001)) {
            ++values_error_counter;
          }
        }
      }

      ASSERT_EQ(values_error_counter, 0);
      const float correct_rate =
          (bigger_score_counter * 1.0f) / MAX_CAPACITY;
      std::cout << std::setprecision(3) << "[Round " << round << "]"
                << "correct_rate=" << correct_rate << std::endl;
      ASSERT_GE(max_key, expected_max_key);
      ASSERT_GE(correct_rate, EXPECTED_CORRECT_RATE);
    }
  }
}

void test_dynamic_rehash_on_multi_threads(size_t max_hbm_for_vectors,
                                          int key_start = 0) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = 4 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024UL * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 256UL;
  constexpr uint64_t THREAD_N = 8UL;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  auto options = make_options(INIT_CAPACITY, MAX_CAPACITY, max_hbm_for_vectors,
                             key_start, BUCKET_MAX_SIZE);
  options.max_load_factor = 0.50f;
  options.api_lock = true;

  auto table = std::make_shared<Table>();
  table->init(options);

  auto worker_function = [&table, KEY_NUM, options](int task_n) {
    auto device_id_env = std::getenv("HKV_TEST_DEVICE");
    int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
    HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                    "aclrtSetDevice failed");
    StreamGuard guard;
    guard.create();
    std::vector<K> h_keys(KEY_NUM);
    std::vector<V> h_vectors(KEY_NUM * DIM);
    std::vector<uint8_t> h_found(KEY_NUM);
    size_t current_capacity = table->capacity();

    DeviceArray<K> d_keys;
    DeviceArray<V> d_vectors;
    DeviceArray<bool> d_found;
    d_keys.alloc(KEY_NUM);
    d_vectors.alloc(KEY_NUM * DIM);
    d_found.alloc(KEY_NUM);

    while (table->capacity() < MAX_CAPACITY) {
      create_random_keys<K, S, V, DIM>(h_keys.data(), nullptr, h_vectors.data(),
                                       KEY_NUM);
      d_keys.copy_from_host(h_keys, guard.stream);
      d_vectors.copy_from_host(h_vectors, guard.stream);
      d_found.memset(0, guard.stream);

      find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), nullptr,
                              d_vectors.get(), options.dim, guard.stream);

      d_vectors.memset(0, guard.stream);
      find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                         d_found.get(), nullptr, guard.stream);
      copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
      d_keys.copy_to_host(&h_keys, KEY_NUM, guard.stream);
      d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);

      size_t found_num = 0;
      for (size_t i = 0; i < KEY_NUM; ++i) {
        if (h_found[i]) {
          ++found_num;
          for (size_t j = 0; j < options.dim; ++j) {
            ASSERT_EQ(h_vectors[i * options.dim + j],
                      static_cast<float>(h_keys[i] * 0.00001));
          }
        }
      }
      ASSERT_EQ(found_num, KEY_NUM);
      if (task_n == 0 && current_capacity != table->capacity()) {
        std::cout << "[test_dynamic_rehash_on_multi_threads] The capacity "
                     "changed from "
                  << current_capacity << " to " << table->capacity()
                  << std::endl;
        current_capacity = table->capacity();
      }
      ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < THREAD_N; ++i) {
    threads.emplace_back(worker_function, static_cast<int>(i));
  }
  for (auto& th : threads) {
    th.join();
  }
}

void test_basic_for_cpu_io(int key_start = 0) {
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  auto options = make_options(INIT_CAPACITY, INIT_CAPACITY, 0, key_start);
  options.io_by_cpu = true;
  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(), nullptr,
                                   KEY_NUM);

  StreamGuard guard;
  guard.create();
  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_found.alloc(KEY_NUM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.memset(1, guard.stream);
  d_found.memset(0, guard.stream);

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);

  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  d_vectors.memset(2, guard.stream);
  table->assign(KEY_NUM, d_keys.get(), d_vectors.get(), d_scores.get(),
                guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_found.get(), nullptr, guard.stream);
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  size_t found_num = 0;
  for (size_t i = 0; i < KEY_NUM; ++i) {
    if (h_found[i]) {
      ++found_num;
    }
  }
  ASSERT_EQ(found_num, KEY_NUM);

  table->accum_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(),
                         d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  table->erase(KEY_NUM >> 1, d_keys.get(), guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM >> 1);

  table->clear(guard.stream);
  ASSERT_EQ(table->size(guard.stream), 0);

  find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  const size_t dump_counter =
      table->export_batch(table->capacity(), 0, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ASSERT_EQ(dump_counter, KEY_NUM);
}

void test_find_or_insert_multi_threads(size_t max_hbm_for_vectors,
                                       const float batch_0_ratio,
                                       const float batch_1_ratio,
                                       bool capacity_silent = true) {
  constexpr uint64_t THREAD_N = 64UL;
  const uint64_t batch_0_size =
      static_cast<uint64_t>(THREAD_N * batch_0_ratio);
  const uint64_t batch_1_size =
      static_cast<uint64_t>(THREAD_N * batch_1_ratio);
  constexpr uint64_t INIT_CAPACITY = 32 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = 128 * 1024 * 1024UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  auto options =
      make_options(INIT_CAPACITY, MAX_CAPACITY, max_hbm_for_vectors, 0,
                  BUCKET_MAX_SIZE);
  options.max_load_factor = 0.50f;
  options.api_lock = true;

  auto table = std::make_shared<Table>();
  table->init(options);

  // assert every key is different
  auto worker1 = [&table, KEY_NUM, options, capacity_silent](int batch, int task_n) {
    auto device_id_env = std::getenv("HKV_TEST_DEVICE");
    int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
    HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                    "aclrtSetDevice failed");
    StreamGuard guard;
    guard.create();
    size_t current_capacity = table->capacity();
    std::vector<K> h_keys(KEY_NUM);
    std::vector<V> h_vectors(KEY_NUM * DIM);
    std::vector<uint8_t> h_found(KEY_NUM);
    DeviceArray<K> d_keys;
    DeviceArray<V> d_vectors;
    DeviceArray<bool> d_found;
    d_keys.alloc(KEY_NUM);
    d_vectors.alloc(KEY_NUM * DIM);
    d_found.alloc(KEY_NUM);

    create_random_keys<K, S, V, DIM>(h_keys.data(), nullptr, h_vectors.data(),
                                     KEY_NUM);
    d_keys.copy_from_host(h_keys, guard.stream);
    d_vectors.copy_from_host(h_vectors, guard.stream);
    d_found.memset(0, guard.stream);

    table->assign(KEY_NUM, d_keys.get(), d_vectors.get(), nullptr,
                  guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                       d_found.get(), nullptr, guard.stream);
    copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
    size_t found_num = 0;
    for (size_t i = 0; i < KEY_NUM; ++i) {
      if (h_found[i]) {
        ++found_num;
      }
    }
    ASSERT_EQ(found_num, 0);

    find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), nullptr,
                            d_vectors.get(), options.dim, guard.stream);
    d_vectors.memset(0, guard.stream);
    find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                       d_found.get(), nullptr, guard.stream);
    d_keys.copy_to_host(&h_keys, KEY_NUM, guard.stream);
    copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
    d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);

    found_num = 0;
    thread_local bool print_unequal{false};
    thread_local uint64_t err_times{0};
    for (size_t i = 0; i < KEY_NUM; ++i) {
      if (h_found[i]) {
        ++found_num;
        for (size_t j = 0; j < options.dim; ++j) {
          if (batch == 2) {
            if (h_vectors[i * options.dim + j] !=
                static_cast<float>(h_keys[i] * 0.00001)) {
              if (!print_unequal) {
                std::cout << " [Thread " << task_n << "]\t";
                UNEQUAL_EXPR(h_vectors[i * options.dim + j],
                             static_cast<float>(h_keys[i] * 0.00001));
                print_unequal = true;
              }
              err_times += 1;
            }
          } else {
            ASSERT_EQ(h_vectors[i * options.dim + j],
                      static_cast<float>(h_keys[i] * 0.00001));
          }
        }
      }
    }

    bool print_thread_id{false};
    if (batch == 0 || batch == 1) {
      ASSERT_EQ(found_num, KEY_NUM);
      ASSERT_EQ(err_times, 0);
    } else if (found_num != KEY_NUM || err_times != 0) {
      std::cout << " [Thread " << task_n << "]\t"
                << "Number of keys(insert/found/error) : " << "(" << KEY_NUM
                << "/" << found_num << "/" << err_times << ") \t";
      print_thread_id = true;
    }
    if (current_capacity != table->capacity() && !capacity_silent) {
      if (!print_thread_id) {
        std::cout << " [Thread " << task_n << "]\t";
      }
      std::cout << "The capacity changed from " << current_capacity << " to "
                << table->capacity() << std::endl;
    } else if (print_thread_id) {
      std::cout << std::endl;
    }
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  };

  auto worker2 = [&table, KEY_NUM, options, capacity_silent](int batch, int task_n) {
    auto device_id_env = std::getenv("HKV_TEST_DEVICE");
    int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
    HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                    "aclrtSetDevice failed");
    StreamGuard guard;
    guard.create();
    size_t current_capacity = table->capacity();
    std::vector<K> h_keys(KEY_NUM);
    std::vector<V> h_vectors(KEY_NUM * DIM);
    std::vector<uint8_t> h_found(KEY_NUM);
    DeviceArray<K> d_keys;
    DeviceArray<V> d_vectors;
    DeviceArray<V> d_new_vectors;
    DeviceArray<bool> d_found;
    d_keys.alloc(KEY_NUM);
    d_vectors.alloc(KEY_NUM * DIM);
    d_new_vectors.alloc(KEY_NUM * DIM);
    d_found.alloc(KEY_NUM);

    create_random_keys<K, S, V, DIM>(h_keys.data(), nullptr, h_vectors.data(),
                                     KEY_NUM);
    d_keys.copy_from_host(h_keys, guard.stream);
    d_vectors.copy_from_host(h_vectors, guard.stream);
    d_found.memset(0, guard.stream);
    d_new_vectors.memset(2, guard.stream);

    find_or_insert_safe(table.get(), KEY_NUM, d_keys.get(), nullptr,
                            d_vectors.get(), options.dim, guard.stream);
    table->assign(KEY_NUM, d_keys.get(), d_new_vectors.get(), nullptr,
                  guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));

    d_vectors.memset(0, guard.stream);
    find_into_contiguous(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                       d_found.get(), nullptr, guard.stream);
    d_keys.copy_to_host(&h_keys, KEY_NUM, guard.stream);
    copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
    d_vectors.copy_to_host(&h_vectors, KEY_NUM * DIM, guard.stream);

    size_t found_num = 0;
    thread_local bool print_unequal{false};
    thread_local uint64_t err_times{0};
    const uint32_t i_value = 0x2020202;
    const float expected_memset_value =
        *(reinterpret_cast<const float*>(&i_value));
    for (size_t i = 0; i < KEY_NUM; ++i) {
      if (h_found[i]) {
        ++found_num;
        for (size_t j = 0; j < options.dim; ++j) {
          if (batch == 2) {
            if (h_vectors[i * options.dim + j] != expected_memset_value) {
              if (!print_unequal) {
                std::cout << " [Thread " << task_n << "]\t";
                UNEQUAL_EXPR(h_vectors[i * options.dim + j],
                             expected_memset_value);
                print_unequal = true;
              }
              err_times += 1;
            }
          } else {
            ASSERT_EQ(h_vectors[i * options.dim + j], expected_memset_value);
          }
        }
      }
    }

    bool print_thread_id{false};
    if (batch == 0 || batch == 1) {
      ASSERT_EQ(found_num, KEY_NUM);
      ASSERT_EQ(err_times, 0);
    } else if (found_num != KEY_NUM || err_times != 0) {
      std::cout << " [Thread " << task_n << "]\t"
                << "Number of keys(insert/found/error) : " << "(" << KEY_NUM
                << "/" << found_num << "/" << err_times << ") \t";
      print_thread_id = true;
    }
    if (current_capacity != table->capacity() && !capacity_silent) {
      if (!print_thread_id) {
        std::cout << " [Thread " << task_n << "]\t";
      }
      std::cout << "The capacity changed from " << current_capacity << " to "
                << table->capacity() << std::endl;
    } else if (print_thread_id) {
      std::cout << std::endl;
    }
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  };

  std::vector<std::thread> threads;
  /* the table is relative idle, and assume there is no eviction */
  int batch = 0;
  std::cout << "[Batch 0] " << batch_0_size << " threads\n";
  for (uint64_t i = 0; i < batch_0_size; i += 2) {
    threads.emplace_back(worker1, batch, static_cast<int>(i));
    threads.emplace_back(worker2, batch, static_cast<int>(i + 1));
  }
  for (auto& th : threads) {
    th.join();
  }
  threads.clear();

  /* test the correct of APIs serially */
  batch = 1;
  std::cout << "[Batch 1] " << batch_1_size << " threads\n";
  for (uint64_t i = batch_0_size; i < batch_0_size + batch_1_size; i += 2) {
    auto th = std::thread(worker1, batch, static_cast<int>(i));
    th.join();
    th = std::thread(worker2, batch, static_cast<int>(i + 1));
    th.join();
  }

  /* eviction may occur */
  batch = 2;
  std::cout << "[Batch 2] " << (THREAD_N - batch_0_size - batch_1_size)
            << " threads\n";
  for (uint64_t i = batch_0_size + batch_1_size; i < THREAD_N; i += 2) {
    threads.emplace_back(worker1, batch, static_cast<int>(i));
    threads.emplace_back(worker2, batch, static_cast<int>(i + 1));
  }
  for (auto& th : threads) {
    th.join();
  }
  if (table->capacity() < MAX_CAPACITY) {
    StreamGuard guard;
    guard.create();
    table->reserve(MAX_CAPACITY, guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  }
  ASSERT_EQ(table->capacity(), MAX_CAPACITY);
}

template <typename KType, typename VType, typename SType, typename Table,
          size_t dim = 64>
void CheckFindOrInsertValues(Table* table, KType* keys, VType* values,
                             SType* scores, size_t len, aclrtStream stream) {
  (void)scores;
  std::map<KType, test_util::ValueArray<VType, dim>> map_before_insert;
  std::map<KType, test_util::ValueArray<VType, dim>> map_after_insert;

  const size_t table_size_before = table->size(stream);
  const size_t cap = table_size_before + len;
  DeviceArray<KType> d_tmp_keys;
  DeviceArray<VType> d_tmp_values;
  DeviceArray<SType> d_tmp_scores;
  d_tmp_keys.alloc(cap);
  d_tmp_values.alloc(cap * dim);
  d_tmp_scores.alloc(cap);
  d_tmp_keys.memset(0, stream);
  d_tmp_values.memset(0, stream);
  d_tmp_scores.memset(0, stream);

  std::vector<KType> h_tmp_keys;
  std::vector<VType> h_tmp_values;
  std::vector<SType> h_tmp_scores;

  const size_t table_size_verify0 =
      table->export_batch(table->capacity(), 0, d_tmp_keys.get(),
                          d_tmp_values.get(), d_tmp_scores.get(), stream);
  ASSERT_EQ(table_size_before, table_size_verify0);
  d_tmp_keys.copy_to_host(&h_tmp_keys, table_size_before, stream);
  d_tmp_values.copy_to_host(&h_tmp_values, table_size_before * dim, stream);
  d_tmp_scores.copy_to_host(&h_tmp_scores, table_size_before, stream);

  for (size_t i = 0; i < table_size_verify0; ++i) {
    auto* vec = reinterpret_cast<test_util::ValueArray<VType, dim>*>(
        h_tmp_values.data() + i * dim);
    map_before_insert[h_tmp_keys[i]] = *vec;
  }

  auto start = std::chrono::steady_clock::now();
  find_or_insert_safe(table, len, keys, nullptr, values, dim, stream);
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  const float dur = static_cast<float>(diff.count());
  (void)dur;

  const size_t table_size_after = table->size(stream);
  const size_t table_size_verify1 =
      table->export_batch(table->capacity(), 0, d_tmp_keys.get(),
                          d_tmp_values.get(), d_tmp_scores.get(), stream);
  ASSERT_EQ(table_size_verify1, table_size_after);

  d_tmp_keys.copy_to_host(&h_tmp_keys, table_size_after, stream);
  d_tmp_values.copy_to_host(&h_tmp_values, table_size_after * dim, stream);
  d_tmp_scores.copy_to_host(&h_tmp_scores, table_size_after, stream);

  for (int64_t i = static_cast<int64_t>(table_size_after) - 1; i >= 0; --i) {
    auto* vec = reinterpret_cast<test_util::ValueArray<VType, dim>*>(
        h_tmp_values.data() + static_cast<size_t>(i) * dim);
    map_after_insert[h_tmp_keys[static_cast<size_t>(i)]] = *vec;
  }

  size_t value_diff_cnt = 0;
  for (auto& it : map_after_insert) {
    test_util::ValueArray<VType, dim>& vec = map_after_insert.at(it.first);
    for (size_t j = 0; j < dim; ++j) {
      if (vec[j] != static_cast<float>(it.first * 0.00001)) {
        ++value_diff_cnt;
        break;
      }
    }
  }
  ASSERT_EQ(value_diff_cnt, 0);
  std::cout << "Check find_or_insert behavior got "
            << "value_diff_cnt: " << value_diff_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << std::endl;
}

void test_find_or_insert_values_check(size_t max_hbm_for_vectors) {
  const size_t U = 524288;
  const size_t init_capacity = 1024;
  const size_t B = 524288 + 13;
  constexpr size_t dim = 64;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  TableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = GB(max_hbm_for_vectors);
  set_simd_value_move_enabled(max_hbm_for_vectors == 0);
  opt.dim = dim;

  StreamGuard guard;
  guard.create();
  auto table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<K, V, S> data_buffer;
  data_buffer.reserve(B, dim, guard.stream);

  size_t offset = 0;
  S score = 0;
  (void)offset;
  (void)score;
  for (int i = 0; i < 20; ++i) {
    create_random_keys<K, S, V, dim>(data_buffer.keys_ptr(false),
                                     data_buffer.scores_ptr(false),
                                     data_buffer.values_ptr(false),
                                     static_cast<int>(B), B * 16);
    data_buffer.sync_data(true, guard.stream);

    CheckFindOrInsertValues<K, V, S, Table, dim>(
        table.get(), data_buffer.keys_ptr(), data_buffer.values_ptr(),
        data_buffer.scores_ptr(), B, guard.stream);

    offset += B;
    score += 1;
  }
}

}  // namespace

TEST(FindOrInsertTest, test_export_batch_if) {
  test_export_batch_if(16);
  test_export_batch_if(0, 31);
}

TEST(FindOrInsertTest, test_find_or_insert_multi_threads) {
  test_find_or_insert_multi_threads(16, 0.25f, 0.125f);
  test_find_or_insert_multi_threads(16, 0.375f, 0.125f);
  test_find_or_insert_multi_threads(0, 0.25f, 0.125f);
  test_find_or_insert_multi_threads(0, 0.375f, 0.125f);
}

TEST(FindOrInsertTest, test_value_type_hbm_mode) {
  test_value_type_hbm_mode<int8_t, 64>();
  test_value_type_hbm_mode<int8_t, 256>();
  test_value_type_hbm_mode<int8_t, 512>();

  test_value_type_hbm_mode<uint8_t, 63>();
  test_value_type_hbm_mode<uint8_t, 255>();
  test_value_type_hbm_mode<uint8_t, 511>();

  test_value_type_hbm_mode<int16_t, 32>();
  test_value_type_hbm_mode<int16_t, 128>();
  test_value_type_hbm_mode<int16_t, 256>();

  test_value_type_hbm_mode<int, 16>();
  test_value_type_hbm_mode<int, 64>();
  test_value_type_hbm_mode<float, 128>();

  test_value_type_hbm_mode<int64_t, 31>();
  test_value_type_hbm_mode<double, 63>();
}

TEST(FindOrInsertTest, test_basic) {
  test_basic(16, 61);
  test_basic(0);
}

TEST(FindOrInsertTest, test_basic_when_full) {
  test_basic_when_full(16);
  test_basic_when_full(0, 41);
}

TEST(FindOrInsertTest, test_erase_if_pred) {
  test_erase_if_pred(16);
  test_erase_if_pred(0, 17);
}

TEST(FindOrInsertTest, test_rehash) {
  test_rehash(16);
  test_rehash(0, 22);
}

TEST(FindOrInsertTest, test_rehash_on_big_batch) {
  test_rehash_on_big_batch(16, 37);
  test_rehash_on_big_batch(0);
}

TEST(FindOrInsertTest, test_dynamic_rehash_on_multi_threads) {
  test_dynamic_rehash_on_multi_threads(16, 22);
  test_dynamic_rehash_on_multi_threads(0);
}

TEST(FindOrInsertTest, test_basic_for_cpu_io) {
  test_basic_for_cpu_io(45);
  test_basic_for_cpu_io();
}

TEST(FindOrInsertTest, test_evict_strategy_lru_basic) {
  test_evict_strategy_lru_basic(16);
  test_evict_strategy_lru_basic(0, 44);
}

TEST(FindOrInsertTest, test_evict_strategy_lfu_basic) {
  test_evict_strategy_lfu_basic(16, 34);
  test_evict_strategy_lfu_basic(0);
}

TEST(FindOrInsertTest, test_evict_strategy_epochlru_basic) {
  test_evict_strategy_epochlru_basic(16, 41);
  test_evict_strategy_epochlru_basic(0);
}

TEST(FindOrInsertTest, test_evict_strategy_epochlfu_basic) {
  test_evict_strategy_epochlfu_basic(16, 42);
  test_evict_strategy_epochlfu_basic(0);
}

TEST(FindOrInsertTest, test_evict_strategy_customized_basic) {
  test_evict_strategy_customized_basic(16);
  test_evict_strategy_customized_basic(0, 43);
}

TEST(FindOrInsertTest, test_evict_strategy_customized_advanced) {
  test_evict_strategy_customized_advanced(16, 54);
  test_evict_strategy_customized_advanced(0);
}

TEST(FindOrInsertTest, test_assign_advanced_on_epochlfu) {
  test_assign_advanced_on_epochlfu(16);
}

TEST(FindOrInsertTest, test_evict_strategy_customized_correct_rate) {
  // TODO: after blossom CI issue is resolved, the skip logic.
  const bool skip_hmem_check = (nullptr != std::getenv("IS_BLOSSOM_CI"));
  test_evict_strategy_customized_correct_rate(16);
  if (!skip_hmem_check) {
    test_evict_strategy_customized_correct_rate(0);
  } else {
    std::cout << "The HMEM check is skipped in blossom CI!" << std::endl;
  }
}

TEST(FindOrInsertTest, test_find_or_insert_values_check) {
  test_find_or_insert_values_check(16);
  // TODO: Add back when diff error issue fixed in hybrid mode.
  test_find_or_insert_values_check(0);
}
