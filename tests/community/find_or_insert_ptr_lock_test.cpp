/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
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

template <class Table>
void find_or_insert_safe_ptr(Table* table, uint64_t key_num, K* d_keys,
                             S* d_scores, V* d_vectors, uint64_t dim,
                             aclrtStream stream) {
  DeviceArray<V*> d_vectors_ptr;
  DeviceArray<bool> d_found;
  DeviceArray<K*> d_key_ptrs;
  d_vectors_ptr.alloc(key_num);
  d_found.alloc(key_num);
  d_key_ptrs.alloc(key_num);

  table->find_or_insert(key_num, d_keys, d_vectors_ptr.get(), d_found.get(),
                        d_scores, stream, true, false, d_key_ptrs.get());
  read_or_write_ptr(d_vectors_ptr.get(), d_vectors, d_found.get(), dim, key_num,
                    stream);
  /// TODO:check the d_found
  table->unlock_keys(key_num, d_key_ptrs.get(), d_keys, d_found.get(), stream);
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

  find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
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

  find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
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
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.memset(1, guard.stream);

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);
  find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  const auto total_size_after_insert = table->size(guard.stream);
  table->erase(KEY_NUM, d_keys.get(), guard.stream);
  ASSERT_EQ(table->size(guard.stream), 0);
  find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
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

  find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
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

    find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
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
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(),
                                   h_vectors.data(), KEY_NUM);

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * DIM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.copy_from_host(h_vectors, guard.stream);

  find_or_insert_safe_ptr(table.get(), INIT_KEY_NUM, d_keys.get(),
                          d_scores.get(), d_vectors.get(), options.dim,
                          guard.stream);
  ASSERT_EQ(table->size(guard.stream), INIT_KEY_NUM);
  ASSERT_EQ(table->capacity(), INIT_CAPACITY * 2);

  find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), d_scores.get(),
                          d_vectors.get(), options.dim, guard.stream);
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  ASSERT_EQ(table->capacity(), KEY_NUM * 4);
  export_and_verify_all(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                     d_scores.get(), guard.stream);
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
  find_or_insert_safe_ptr(table.get(), KEY_NUM, d_keys.get(), nullptr,
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
    find_or_insert_safe_ptr(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), BASE_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), TEST_KEY_NUM, d_keys_temp.get(),
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
    find_or_insert_safe_ptr(table.get(), BASE_KEY_NUM, d_keys.get(),
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
    find_or_insert_safe_ptr(table.get(), TEST_KEY_NUM, d_keys.get(),
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
    find_or_insert_safe_ptr(table.get(), BASE_KEY_NUM, d_keys.get(),
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
    find_or_insert_safe_ptr(table.get(), TEST_KEY_NUM, d_keys.get(),
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
        find_or_insert_safe_ptr(table.get(), BATCH_SIZE, d_keys_temp.get(),
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

}  // namespace

TEST(FindOrInsertPtrLockTest, test_export_batch_if) {
  test_export_batch_if(16);
  test_export_batch_if(0, 33);
}

TEST(FindOrInsertPtrLockTest, test_basic) {
  test_basic(16, 3);
  test_basic(0);
}

TEST(FindOrInsertPtrLockTest, test_basic_when_full) {
  test_basic_when_full(16, 4);
  test_basic_when_full(0);
}

TEST(FindOrInsertPtrLockTest, test_erase_if_pred) {
  test_erase_if_pred(16, 14);
  test_erase_if_pred(0);
}

TEST(FindOrInsertPtrLockTest, test_rehash) {
  test_rehash(16);
  test_rehash(0, 42);
}

TEST(FindOrInsertPtrLockTest, test_rehash_on_big_batch) {
  test_rehash_on_big_batch(16, 11);
  test_rehash_on_big_batch(0);
}

TEST(FindOrInsertPtrLockTest, test_evict_strategy_lru_basic) {
  test_evict_strategy_lru_basic(16);
  test_evict_strategy_lru_basic(0, 18);
}

TEST(FindOrInsertPtrLockTest, test_evict_strategy_lfu_basic) {
  test_evict_strategy_lfu_basic(16, 29);
  test_evict_strategy_lfu_basic(0);
}

TEST(FindOrInsertPtrLockTest, test_evict_strategy_epochlru_basic) {
  test_evict_strategy_epochlru_basic(16, 45);
  test_evict_strategy_epochlru_basic(0);
}

TEST(FindOrInsertPtrLockTest, test_evict_strategy_epochlfu_basic) {
  test_evict_strategy_epochlfu_basic(16);
  test_evict_strategy_epochlfu_basic(0, 59);
}

TEST(FindOrInsertPtrLockTest, test_evict_strategy_customized_basic) {
  test_evict_strategy_customized_basic(16, 38);
  test_evict_strategy_customized_basic(0);
}

TEST(FindOrInsertPtrLockTest, test_evict_strategy_customized_advanced) {
  test_evict_strategy_customized_advanced(16);
  test_evict_strategy_customized_advanced(0, 25);
}

TEST(FindOrInsertPtrLockTest, test_evict_strategy_customized_correct_rate) {
  // TODO: after blossom CI issue is resolved, the skip logic.
  test_evict_strategy_customized_correct_rate(16, 16);
  test_evict_strategy_customized_correct_rate(0);
}

// Turn on to verify that it can't deal with multi-threads cases
// TEST(FindOrInsertPtrLockTest, test_find_or_insert_multi_threads) {}

// TEST(FindOrInsertPtrLockTest, test_dynamic_rehash_on_multi_threads) {}

// Turn on to verify that it can't deal with small capacity case
// TEST(FindOrInsertPtrLockTest, test_find_or_insert_values_check) {
// TODO: Add back when diff error issue fixed in hybrid mode.
// }
