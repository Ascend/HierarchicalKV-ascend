/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cstdlib>
#include <cstring>
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

template <class KType, class VType, class SType>
struct EraseIfPredFunctorV2 {
  KType pattern;
  SType threshold;
  EraseIfPredFunctorV2(KType pattern, SType threshold)
      : pattern(pattern), threshold(threshold) {}

  __forceinline__ __simt_callee__ bool operator()(const KType& key,
                                                  const __gm__ VType* value,
                                                  const SType& score,
                                                  int32_t group_size) {
    (void)value;
    (void)group_size;
    /* evaluate key, score and value. */
    return (((key & 0x7f) > pattern) && (score > threshold));
  }
};

template <class KType, class VType, class SType>
struct EraseIfPredFunctorV3 {
  KType pattern;
  SType threshold;
  uint32_t dim;
  EraseIfPredFunctorV3(KType pattern, SType threshold)
      : pattern(pattern), threshold(threshold), dim(0) {}

  __forceinline__ __simt_callee__ bool operator()(const KType& key,
                                                  const __gm__ VType* value,
                                                  const SType& score,
                                                  int32_t group_size) {
    (void)group_size;
    /* evaluate key, score and value. */
    bool pred = score < threshold;
    for (uint32_t i = 0; pred && i < dim; ++i) {
      if (value[i] != static_cast<VType>(key * 0.00001f)) {
        pred = false;
      }
    }
    return pred;
  }
};

enum class EraseIfVersion { V1, V2, V3 };

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
                         size_t max_hbm_for_vectors,
                         int reserved_key_start_bit, size_t dim = DIM,
                         size_t max_bucket_size = 128) {
  TableOptions options{};
  options.init_capacity = init_capacity;
  options.max_capacity = max_capacity;
  options.dim = dim;
  options.max_bucket_size = max_bucket_size;
  options.max_hbm_for_vectors = GB(max_hbm_for_vectors);
  options.reserved_key_start_bit = reserved_key_start_bit;
  return options;
}

template <class T>
void copy_device_to_host_raw(T* device, std::vector<T>* host, size_t n,
                         aclrtStream stream) {
  if (n == 0) {
    return;
  }
  host->assign(n, T{});
  ACL_CHECK(aclrtMemcpyAsync(host->data(), sizeof(T) * n, device,
                             sizeof(T) * n, ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
}

void copy_bool_to_host(bool* device, std::vector<uint8_t>* host, size_t n,
                    aclrtStream stream) {
  host->assign(n, uint8_t{0});
  ACL_CHECK(aclrtMemcpyAsync(host->data(), sizeof(uint8_t) * n, device,
                             sizeof(bool) * n, ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
}

template <class VType>
VType byte_pattern_value() {
  uint32_t pattern = 0x02020202;
  VType value{};
  std::memcpy(&value, &pattern, std::min(sizeof(VType), sizeof(pattern)));
  return value;
}

template <typename T, std::size_t N>
std::array<T, N> range(const T start) {
  std::array<T, N> values{};
  for (std::size_t i = 0; i < N; ++i) {
    values[i] = start + static_cast<T>(i);
  }
  return values;
}

template <class VType>
void VerifyValuesUsingKeys(const std::vector<K>& keys,
                           const std::vector<VType>& values, size_t n,
                           size_t dim) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(values[i * dim + j], static_cast<VType>(keys[i] * 0.00001));
    }
  }
}

template <class VType>
void verify_found_values_and_scores(const std::vector<K>& keys,
                                const std::vector<S>& scores,
                                const std::vector<VType>& values,
                                const std::vector<uint8_t>& found,
                                size_t expected_found, size_t dim,
                                bool verify_scores) {
  size_t found_num = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (!found[i]) {
      continue;
    }
    ++found_num;
    if (verify_scores) {
      ASSERT_EQ(scores[i], keys[i]);
    }
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(values[i * dim + j], static_cast<VType>(keys[i] * 0.00001));
    }
  }
  ASSERT_EQ(found_num, expected_found);
}

template <class Table, class VType>
void export_to_host(Table* table, size_t expected_count, K* d_keys,
                  VType* d_values, S* d_scores, std::vector<K>* keys,
                  std::vector<VType>* values, std::vector<S>* scores,
                  aclrtStream stream, size_t dim) {
  const size_t exported =
      table->export_batch(table->capacity(), 0, d_keys, d_values, d_scores,
                          stream);
  ASSERT_EQ(exported, expected_count);
  copy_device_to_host_raw(d_keys, keys, expected_count, stream);
  copy_device_to_host_raw(d_values, values, expected_count * dim, stream);
  copy_device_to_host_raw(d_scores, scores, expected_count, stream);
}

template <class Table, class VType>
void FindAndVerify(Table* table, size_t n, K* d_keys, VType* d_values,
                   S* d_scores, bool* d_found, std::vector<K>* h_keys,
                   std::vector<S>* h_scores, std::vector<VType>* h_values,
                   std::vector<uint8_t>* h_found, aclrtStream stream,
                   size_t dim, size_t expected_found, bool verify_scores) {
  table->find(n, d_keys, d_values, d_found, verify_scores ? d_scores : nullptr,
              stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  copy_device_to_host_raw(d_keys, h_keys, n, stream);
  copy_bool_to_host(d_found, h_found, n, stream);
  if (verify_scores) {
    copy_device_to_host_raw(d_scores, h_scores, n, stream);
  }
  copy_device_to_host_raw(d_values, h_values, n * dim, stream);
  verify_found_values_and_scores(*h_keys, *h_scores, *h_values, *h_found,
                             expected_found, dim, verify_scores);
}

template <class Table>
void ContainsAndVerify(Table* table, size_t n, K* d_keys, bool* d_found,
                       size_t expected_found, aclrtStream stream) {
  table->contains(n, d_keys, d_found, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  std::vector<uint8_t> h_found;
  copy_bool_to_host(d_found, &h_found, n, stream);
  size_t contains_num = 0;
  for (uint8_t flag : h_found) {
    if (flag) {
      ++contains_num;
    }
  }
  ASSERT_EQ(contains_num, expected_found);
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

void test_basic_common(size_t max_hbm_for_vectors, bool without_rehash) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128;
  constexpr uint64_t NUM_OF_BUCKETS_PER_ALLOC = 2048;
  const uint64_t init_capacity =
      without_rehash
          ? (64 * 1024 * 1024UL - (NUM_OF_BUCKETS_PER_ALLOC * BUCKET_MAX_SIZE) +
             1)
          : (64 * 1024 * 1024UL - (BUCKET_MAX_SIZE + 1));
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(init_capacity, init_capacity,
                                     max_hbm_for_vectors, 2, DIM,
                                     BUCKET_MAX_SIZE);
  options.num_of_buckets_per_alloc = without_rehash ? NUM_OF_BUCKETS_PER_ALLOC
                                                    : 32;
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(),
                                   h_vectors.data(), KEY_NUM);
  const std::vector<V> h_original_vectors = h_vectors;

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
  d_vectors.copy_from_host(h_original_vectors, guard.stream);

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->bucket_count(), without_rehash ? 522241 : 524287);
  ASSERT_EQ(table->size(guard.stream), 0);

  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  d_vectors.memset(0, guard.stream);
  FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                d_scores.get(), d_found.get(), &h_keys, &h_scores, &h_vectors,
                &h_found, guard.stream, DIM, KEY_NUM, false);
  for (size_t i = 0; i < KEY_NUM; ++i) {
    ASSERT_EQ(h_scores[i], h_keys[i]);
  }
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(), KEY_NUM,
                    guard.stream);

  d_new_vectors.memset(2, guard.stream);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_new_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  d_new_vectors.memset(0, guard.stream);
  table->find(KEY_NUM, d_keys.get(), d_new_vectors.get(), d_found.get(),
              nullptr, guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  copy_device_to_host_raw(d_new_vectors.get(), &h_vectors, KEY_NUM * DIM,
                      guard.stream);
  size_t found_num = 0;
  const V byte_value = byte_pattern_value<V>();
  for (size_t i = 0; i < KEY_NUM; ++i) {
    if (h_found[i]) {
      ++found_num;
    }
    for (size_t j = 0; j < DIM; ++j) {
      ASSERT_EQ(h_vectors[i * DIM + j], byte_value);
    }
  }
  ASSERT_EQ(found_num, KEY_NUM);
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(), KEY_NUM,
                    guard.stream);

  table->accum_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(),
                         d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  table->erase(KEY_NUM >> 1, d_keys.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM >> 1);

  table->clear(guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), 0);

  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  d_vectors.memset(0, guard.stream);
  d_scores.memset(0, guard.stream);
  FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                d_scores.get(), d_found.get(), &h_keys, &h_scores, &h_vectors,
                &h_found, guard.stream, DIM, KEY_NUM, true);
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(), KEY_NUM,
                    guard.stream);

  d_keys.memset(0, guard.stream);
  d_scores.memset(0, guard.stream);
  d_vectors.memset(0, guard.stream);
  export_to_host(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
               d_scores.get(), &h_keys, &h_vectors, &h_scores, guard.stream,
               DIM);
  for (size_t i = 0; i < KEY_NUM; ++i) {
    ASSERT_EQ(h_scores[i], h_keys[i]);
  }
  VerifyValuesUsingKeys(h_keys, h_vectors, KEY_NUM, DIM);
}

template <typename VType>
void test_find_using_pipeline(int dim, bool load_scores) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = 128 * 1024UL;
  constexpr uint64_t KEY_NUM = 128UL;
  using Table = HashTable<K, VType, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY, 16, 1, dim,
                                     BUCKET_MAX_SIZE);
  options.num_of_buckets_per_alloc = 2;
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<VType> h_vectors(KEY_NUM * dim);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_random_keys<K, S, VType>(options.dim, h_keys.data(), h_scores.data(),
                                  h_vectors.data(), KEY_NUM);

  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<VType> d_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * dim);
  d_found.alloc(KEY_NUM);
  d_keys.copy_from_host(h_keys, guard.stream);
  d_scores.copy_from_host(h_scores, guard.stream);
  d_vectors.copy_from_host(h_vectors, guard.stream);

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);

  d_vectors.memset(0, guard.stream);
  d_scores.memset(0, guard.stream);
  table->find(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(),
              load_scores ? d_scores.get() : nullptr, guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  copy_device_to_host_raw(d_vectors.get(), &h_vectors, KEY_NUM * dim,
                      guard.stream);
  if (load_scores) {
    copy_device_to_host_raw(d_scores.get(), &h_scores, KEY_NUM, guard.stream);
  }
  verify_found_values_and_scores(h_keys, h_scores, h_vectors, h_found, KEY_NUM,
                             dim, load_scores);
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(), KEY_NUM,
                    guard.stream);
}

void test_basic_when_full(size_t max_hbm_for_vectors) {
  constexpr uint64_t INIT_CAPACITY = 1 * 1024 * 1024UL;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY,
                                     max_hbm_for_vectors, 3);
  options.num_of_buckets_per_alloc = 32;
  StreamGuard guard;
  guard.create();

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

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  const uint64_t total_size_after_insert = table->size(guard.stream);
  d_vectors.memset(0, guard.stream);
  FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                d_scores.get(), d_found.get(), &h_keys, &h_scores, &h_vectors,
                &h_found, guard.stream, DIM, total_size_after_insert, false);
  table->erase(KEY_NUM, d_keys.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), 0);
  d_vectors.copy_from_host(h_vectors, guard.stream);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), total_size_after_insert);
}

template <EraseIfVersion EV>
void test_erase_if_pred(size_t max_hbm_for_vectors) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY,
                                     max_hbm_for_vectors, 4);
  options.num_of_buckets_per_alloc = 2;
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys.data(), h_scores.data(), h_vectors.data(), KEY_NUM,
      INIT_CAPACITY);

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

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);

  constexpr K pattern = 100;
  constexpr S threshold = 0;
  size_t erase_num = 0;
  if constexpr (EV == EraseIfVersion::V1) {
    erase_num = table->template erase_if<EraseIfPredFunctor>(
        pattern, threshold, guard.stream);
  } else if constexpr (EV == EraseIfVersion::V2) {
    EraseIfPredFunctorV2<K, V, S> pred(pattern, threshold);
    erase_num = table->erase_if_v2(pred, guard.stream);
  } else {
    EraseIfPredFunctorV3<K, V, S> pred(pattern, threshold);
    pred.dim = options.dim;
    erase_num = table->erase_if_v2(pred, guard.stream);
  }
  ASSERT_EQ(erase_num + table->size(guard.stream), BUCKET_MAX_SIZE);
  d_vectors.memset(0, guard.stream);
  d_scores.memset(0, guard.stream);
  FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                d_scores.get(), d_found.get(), &h_keys, &h_scores, &h_vectors,
                &h_found, guard.stream, DIM, BUCKET_MAX_SIZE - erase_num,
                true);
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(),
                    BUCKET_MAX_SIZE - erase_num, guard.stream);
  table->clear(guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), 0);
}

void test_rehash(size_t max_hbm_for_vectors) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = 4 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = BUCKET_MAX_SIZE * 2;
  constexpr uint64_t TEST_TIMES = 100;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, MAX_CAPACITY,
                                     max_hbm_for_vectors, 5, DIM,
                                     BUCKET_MAX_SIZE);
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
    table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                            d_scores.get(), guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    ASSERT_EQ(table->size(guard.stream), KEY_NUM);
    export_to_host(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys, &h_vectors, &h_scores, guard.stream,
                 DIM);
    table->reserve(MAX_CAPACITY, guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    ASSERT_EQ(table->capacity(), MAX_CAPACITY);
    ASSERT_EQ(table->size(guard.stream), KEY_NUM);
    d_vectors.memset(0, guard.stream);
    FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                  d_scores.get(), d_found.get(), &h_keys, &h_scores,
                  &h_vectors, &h_found, guard.stream, DIM, KEY_NUM,
                  true);
    ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(),
                      KEY_NUM, guard.stream);
    table->clear(guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    ASSERT_EQ(table->size(guard.stream), 0);
  }
}

void test_rehash_on_big_batch(size_t max_hbm_for_vectors) {
  constexpr uint64_t INIT_CAPACITY = 1024;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024;
  constexpr uint64_t INIT_KEY_NUM = 1024;
  constexpr uint64_t KEY_NUM = 2048;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, MAX_CAPACITY,
                                     max_hbm_for_vectors, 6);
  options.num_of_buckets_per_alloc = 8;
  options.max_load_factor = 0.6f;
  StreamGuard guard;
  guard.create();

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

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);
  table->insert_or_assign(INIT_KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), INIT_KEY_NUM);
  ASSERT_EQ(table->capacity(), INIT_CAPACITY * 2);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  ASSERT_EQ(table->capacity(), KEY_NUM * 4);
  export_to_host(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
               d_scores.get(), &h_keys, &h_vectors, &h_scores, guard.stream,
               DIM);
  d_vectors.memset(0, guard.stream);
  FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                d_scores.get(), d_found.get(), &h_keys, &h_scores, &h_vectors,
                &h_found, guard.stream, DIM, KEY_NUM, true);
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(), KEY_NUM,
                    guard.stream);
  table->clear(guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), 0);
}

void test_rehash_on_big_batch_specific(size_t max_hbm_for_vectors) {
  constexpr uint64_t INIT_CAPACITY = 50000;
  constexpr uint64_t MAX_CAPACITY = 100000;
  constexpr uint64_t EXPECTED_MAX_CAPACITY = 65536;
  constexpr uint64_t KEY_NUM = 50000;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, MAX_CAPACITY,
                                     max_hbm_for_vectors, 7);
  options.num_of_buckets_per_alloc = 16;
  options.max_load_factor = 0.6f;
  StreamGuard guard;
  guard.create();

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

  auto table = std::make_unique<Table>();
  table->init(options);
  ASSERT_EQ(table->size(guard.stream), 0);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), nullptr,
                          guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->capacity(), EXPECTED_MAX_CAPACITY);
}

void test_dynamic_rehash_on_multi_threads(size_t max_hbm_for_vectors) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = 4 * 1024UL - BUCKET_MAX_SIZE - 1;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024UL * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 256UL;
  constexpr uint64_t THREAD_N = 8UL;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, MAX_CAPACITY,
                                     max_hbm_for_vectors, 8, DIM,
                                     BUCKET_MAX_SIZE);
  options.num_of_buckets_per_alloc = 16;
  options.max_load_factor = 0.50f;
  options.api_lock = true;
  auto table = std::make_shared<Table>();
  table->init(options);
  ASSERT_EQ(table->bucket_count(), 32);

  auto worker_function = [&table, KEY_NUM, options](int task_n) {
    auto device_id_env = std::getenv("HKV_TEST_DEVICE");
    int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
    HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                    "aclrtSetDevice failed");
    StreamGuard guard;
    guard.create();
    std::vector<K> h_keys(KEY_NUM);
    std::vector<V> h_vectors(KEY_NUM * options.dim);
    std::vector<uint8_t> h_found(KEY_NUM);
    size_t current_capacity = table->capacity();

    DeviceArray<K> d_keys;
    DeviceArray<V> d_vectors;
    DeviceArray<bool> d_found;
    d_keys.alloc(KEY_NUM);
    d_vectors.alloc(KEY_NUM * options.dim);
    d_found.alloc(KEY_NUM);

    while (table->capacity() * 2 < options.max_capacity) {
      create_random_keys<K, S, V, DIM>(h_keys.data(), nullptr,
                                       h_vectors.data(), KEY_NUM);
      d_keys.copy_from_host(h_keys, guard.stream);
      d_vectors.copy_from_host(h_vectors, guard.stream);
      table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), nullptr,
                              guard.stream);
      ACL_CHECK(aclrtSynchronizeStream(guard.stream));
      d_vectors.memset(0, guard.stream);
      table->find(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(),
                  nullptr, guard.stream);
      ACL_CHECK(aclrtSynchronizeStream(guard.stream));
      copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
      copy_device_to_host_raw(d_keys.get(), &h_keys, KEY_NUM, guard.stream);
      copy_device_to_host_raw(d_vectors.get(), &h_vectors, KEY_NUM * options.dim,
                          guard.stream);
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
      ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(),
                        KEY_NUM, guard.stream);
      if (task_n == 0 && current_capacity != table->capacity()) {
        std::cout << "[test_dynamic_rehash_on_multi_threads] The capacity "
                     "changed from "
                  << current_capacity << " to " << table->capacity()
                  << std::endl;
        current_capacity = table->capacity();
      }
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < THREAD_N; ++i) {
    threads.emplace_back(worker_function, static_cast<int>(i));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

void test_export_batch_if(size_t max_hbm_for_vectors) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY,
                                     max_hbm_for_vectors, 9);
  options.num_of_buckets_per_alloc = 2;
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

  S threshold = npu::hkv::host_nano<S>(guard.stream);
  ASSERT_EQ(table->size(guard.stream), 0);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), nullptr,
                          guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  d_vectors.memset(0, guard.stream);
  FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                d_scores.get(), d_found.get(), &h_keys, &h_scores, &h_vectors,
                &h_found, guard.stream, DIM, KEY_NUM, false);
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(),
                    BUCKET_MAX_SIZE, guard.stream);

  constexpr K pattern = 100;
  table->template export_batch_if<ExportIfPredFunctor>(
      pattern, threshold, table->capacity(), 0, d_dump_counter.get(),
      d_keys.get(), d_vectors.get(), d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  std::vector<size_t> h_dump;
  d_dump_counter.copy_to_host(&h_dump, 1, guard.stream);
  copy_device_to_host_raw(d_scores.get(), &h_scores, KEY_NUM, guard.stream);
  size_t expected_export_count = 0;
  for (size_t i = 0; i < KEY_NUM; ++i) {
    if (h_scores[i] > threshold) {
      ++expected_export_count;
    }
  }
  ASSERT_EQ(expected_export_count, h_dump[0]);

  threshold = npu::hkv::host_nano<S>(guard.stream);
  table->template export_batch_if<ExportIfPredFunctor>(
      pattern, threshold, table->capacity(), 0, d_dump_counter.get(),
      d_keys.get(), d_vectors.get(), d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  d_dump_counter.copy_to_host(&h_dump, 1, guard.stream);
  ASSERT_EQ(0, h_dump[0]);
  copy_device_to_host_raw(d_keys.get(), &h_keys, h_dump[0], guard.stream);
  copy_device_to_host_raw(d_scores.get(), &h_scores, h_dump[0], guard.stream);
  copy_device_to_host_raw(d_vectors.get(), &h_vectors, h_dump[0] * DIM,
                      guard.stream);
  for (size_t i = 0; i < h_dump[0]; ++i) {
    ASSERT_GT(h_scores[i], threshold);
  }
  table->clear(guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), 0);
}

void test_basic_for_cpu_io() {
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY, 0, 10);
  options.io_by_cpu = true;
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * DIM, V{});
  std::vector<uint8_t> h_found(KEY_NUM);
  create_random_keys<K, S, V, DIM>(h_keys.data(), h_scores.data(), nullptr,
                                   KEY_NUM);

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
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  d_vectors.memset(2, guard.stream);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  table->find(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(), nullptr,
              guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
  size_t found_num = 0;
  for (uint8_t flag : h_found) {
    if (flag) {
      ++found_num;
    }
  }
  ASSERT_EQ(found_num, KEY_NUM);
  ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(), KEY_NUM,
                    guard.stream);
  table->accum_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(),
                         d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM);
  table->erase(KEY_NUM >> 1, d_keys.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), KEY_NUM >> 1);
  table->clear(guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  ASSERT_EQ(table->size(guard.stream), 0);
  table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  const size_t dump_counter =
      table->export_batch(table->capacity(), 0, d_keys.get(), d_vectors.get(),
                          d_scores.get(), guard.stream);
  ASSERT_EQ(dump_counter, KEY_NUM);
}

template <int Strategy>
void test_evict_strategy_basic(size_t max_hbm_for_vectors) {
  constexpr int RSHIFT_ON_NANO = 20;
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM =
      Strategy == EvictStrategy::kCustomized ? 128UL : 4UL;
  constexpr uint64_t TEMP_KEY_NUM = std::max(TEST_KEY_NUM, BASE_KEY_NUM);
  constexpr uint64_t TEST_TIMES =
      Strategy == EvictStrategy::kLfu ? 1024UL : 128UL;
  using Table = HashTable<K, V, S, Strategy>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY,
                                     max_hbm_for_vectors,
                                     Strategy == EvictStrategy::kLru
                                         ? 11
                                         : Strategy == EvictStrategy::kLfu
                                               ? 12
                                               : Strategy ==
                                                         EvictStrategy::kEpochLru
                                                     ? 13
                                                     : Strategy ==
                                                               EvictStrategy::
                                                                   kEpochLfu
                                                           ? 14
                                                           : 15);
  options.num_of_buckets_per_alloc =
      Strategy == EvictStrategy::kLfu ? 1 : (Strategy == EvictStrategy::kLru ? 4 : 8);
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

  constexpr int FREQ_RANGE = 1000;
  if constexpr (Strategy == EvictStrategy::kLfu ||
                Strategy == EvictStrategy::kEpochLfu) {
    create_keys_in_one_buckets_lfu<K, S, V, DIM>(
        h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
        BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
        0x3FFFFFFFFFFFFFFFUL, FREQ_RANGE);
    create_keys_in_one_buckets_lfu<K, S, V, DIM>(
        h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
        TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
        0xFFFFFFFFFFFFFFFDUL, FREQ_RANGE);
  } else {
    create_keys_in_one_buckets<K, S, V, DIM>(
        h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
        BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
        0x3FFFFFFFFFFFFFFFUL);
    create_keys_in_one_buckets<K, S, V, DIM>(
        h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
        TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFFUL,
        0xFFFFFFFFFFFFFFFDUL);
  }

  if constexpr (Strategy == EvictStrategy::kCustomized) {
    constexpr S BASE_SCORE_START = 1000;
    for (size_t i = 0; i < BASE_KEY_NUM; ++i) {
      h_scores_base[i] = BASE_SCORE_START + i;
    }
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
  } else {
    h_keys_test[2] = h_keys_base[72];
    h_keys_test[3] = h_keys_base[73];
    if constexpr (Strategy == EvictStrategy::kLfu ||
                  Strategy == EvictStrategy::kEpochLfu) {
      h_scores_test[2] = h_keys_base[72] % FREQ_RANGE;
      h_scores_test[3] = h_keys_base[73] % FREQ_RANGE;
    }
    if constexpr (Strategy == EvictStrategy::kEpochLfu) {
      // Simulate overflow of low 32bits.
      h_scores_base[71] =
          static_cast<S>(std::numeric_limits<uint32_t>::max() -
                         static_cast<uint32_t>(1));
      h_keys_test[1] = h_keys_base[71];
      h_scores_test[1] = h_scores_base[71];
    }
    const size_t repeated_start =
        Strategy == EvictStrategy::kEpochLfu ? 1UL : 2UL;
    for (size_t i = repeated_start; i < TEST_KEY_NUM; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        h_vectors_test[i * DIM + j] = h_vectors_base[(70 + i) * DIM + j];
      }
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
    S global_epoch = 1;

    d_keys.copy_from_host(h_keys_base, guard.stream);
    d_scores.copy_from_host(h_scores_base, guard.stream);
    d_vectors.copy_from_host(h_vectors_base, guard.stream);
    const S start_ts = Strategy == EvictStrategy::kEpochLru
                           ? ((npu::hkv::host_nano<S>(guard.stream) >>
                               RSHIFT_ON_NANO) &
                              0xFFFFFFFF)
                           : npu::hkv::host_nano<S>(guard.stream);
    set_epoch_if_needed(table.get(), global_epoch);
    table->insert_or_assign(BASE_KEY_NUM, d_keys.get(), d_vectors.get(),
                            scores_for_insert<Strategy>(d_scores.get()),
                            guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    const S end_ts = Strategy == EvictStrategy::kEpochLru
                         ? ((npu::hkv::host_nano<S>(guard.stream) >>
                             RSHIFT_ON_NANO) &
                            0xFFFFFFFF)
                         : npu::hkv::host_nano<S>(guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream, DIM);

    if constexpr (Strategy == EvictStrategy::kLru) {
      auto sorted = h_scores_temp;
      std::sort(sorted.begin(), sorted.end());
      ASSERT_GE(sorted[0], start_ts);
      ASSERT_LE(sorted[BASE_KEY_NUM - 1], end_ts);
    } else if constexpr (Strategy == EvictStrategy::kEpochLru) {
      auto sorted = h_scores_temp;
      std::sort(sorted.begin(), sorted.end());
      ASSERT_GE(sorted[0], ((global_epoch << 32) | start_ts));
      ASSERT_LE(sorted[BASE_KEY_NUM - 1], ((global_epoch << 32) | end_ts));
    } else if constexpr (Strategy == EvictStrategy::kLfu) {
      for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
        ASSERT_EQ(h_scores_temp[i], h_keys_temp[i] % FREQ_RANGE);
      }
    } else if constexpr (Strategy == EvictStrategy::kEpochLfu) {
      for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
        const S original_score = h_keys_temp[i] == h_keys_base[71]
                                     ? h_scores_base[71]
                                     : h_keys_temp[i] % FREQ_RANGE;
        ASSERT_EQ(h_scores_temp[i],
                  make_expected_score_for_epochlfu<S>(global_epoch,
                                                      original_score));
      }
    } else if constexpr (Strategy == EvictStrategy::kCustomized) {
      auto sorted = h_scores_temp;
      std::sort(sorted.begin(), sorted.end());
      auto expected = range<S, TEMP_KEY_NUM>(1000);
      ASSERT_TRUE(std::equal(sorted.begin(), sorted.end(), expected.begin()));
    }
    VerifyValuesUsingKeys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    ++global_epoch;
    d_keys.copy_from_host(h_keys_test, guard.stream);
    d_scores.copy_from_host(h_scores_test, guard.stream);
    d_vectors.copy_from_host(h_vectors_test, guard.stream);
    const S second_start_ts = Strategy == EvictStrategy::kEpochLru
                                  ? ((npu::hkv::host_nano<S>(guard.stream) >>
                                      RSHIFT_ON_NANO) &
                                     0xFFFFFFFF)
                                  : npu::hkv::host_nano<S>(guard.stream);
    set_epoch_if_needed(table.get(), global_epoch);
    table->insert_or_assign(TEST_KEY_NUM, d_keys.get(), d_vectors.get(),
                            scores_for_insert<Strategy>(d_scores.get()),
                            guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    const S second_end_ts = Strategy == EvictStrategy::kEpochLru
                                ? ((npu::hkv::host_nano<S>(guard.stream) >>
                                    RSHIFT_ON_NANO) &
                                   0xFFFFFFFF)
                                : npu::hkv::host_nano<S>(guard.stream);
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream, DIM);

    if constexpr (Strategy == EvictStrategy::kCustomized) {
      auto sorted = h_scores_temp;
      std::sort(sorted.begin(), sorted.end());
      auto expected = range<S, TEST_KEY_NUM>(1128);
      ASSERT_TRUE(std::equal(sorted.begin(), sorted.end(), expected.begin()));
    } else if constexpr (Strategy == EvictStrategy::kLru) {
      std::array<S, TEMP_KEY_NUM> refreshed_scores{};
      size_t refreshed_count = 0;
      for (size_t i = 0; i < TEMP_KEY_NUM; ++i) {
        if (std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]) !=
            h_keys_test.end()) {
          ASSERT_GT(h_scores_temp[i], BUCKET_MAX_SIZE);
          refreshed_scores[refreshed_count++] = h_scores_temp[i];
        } else {
          ASSERT_LE(h_scores_temp[i], second_start_ts);
        }
      }
      std::sort(refreshed_scores.begin(),
                refreshed_scores.begin() + refreshed_count);
      ASSERT_GE(refreshed_scores[0], second_start_ts);
      ASSERT_LE(refreshed_scores[refreshed_count - 1], second_end_ts);
    } else if constexpr (Strategy == EvictStrategy::kEpochLru) {
      std::vector<S> refreshed_scores;
      refreshed_scores.reserve(TEMP_KEY_NUM);
      for (size_t i = 0; i < TEMP_KEY_NUM; ++i) {
        if (std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i]) !=
            h_keys_test.end()) {
          ASSERT_GE(h_scores_temp[i], ((global_epoch << 32) | second_start_ts));
          refreshed_scores.push_back(h_scores_temp[i]);
        } else {
          ASSERT_LE(h_scores_temp[i], ((global_epoch << 32) | second_start_ts));
        }
      }
      std::sort(refreshed_scores.begin(), refreshed_scores.end());
      if (!refreshed_scores.empty()) {
        ASSERT_GE(refreshed_scores[0],
                  ((global_epoch << 32) | second_start_ts));
        ASSERT_LE(refreshed_scores[refreshed_scores.size() - 1],
                  ((global_epoch << 32) | second_end_ts));
      }
    } else if constexpr (Strategy == EvictStrategy::kLfu) {
      for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
        const bool in_base = std::find(h_keys_base.begin(), h_keys_base.end(),
                                       h_keys_temp[i]) != h_keys_base.end();
        const bool in_test = std::find(h_keys_test.begin(), h_keys_test.end(),
                                       h_keys_temp[i]) != h_keys_test.end();
        if (in_base && in_test) {
          ASSERT_EQ(h_scores_temp[i], (h_keys_temp[i] % FREQ_RANGE) * 2);
        } else {
          ASSERT_EQ(h_scores_temp[i], h_keys_temp[i] % FREQ_RANGE);
        }
      }
    } else if constexpr (Strategy == EvictStrategy::kEpochLfu) {
      ASSERT_NE(h_keys_temp.end(),
                std::find(h_keys_temp.begin(), h_keys_temp.end(),
                          h_keys_base[71]));
      for (size_t i = 0; i < BUCKET_MAX_SIZE; ++i) {
        const bool in_base = std::find(h_keys_base.begin(), h_keys_base.end(),
                                       h_keys_temp[i]) != h_keys_base.end();
        const bool in_test = std::find(h_keys_test.begin(), h_keys_test.end(),
                                       h_keys_temp[i]) != h_keys_test.end();
        S original_score = h_keys_temp[i] == h_keys_base[71]
                               ? h_scores_base[71]
                               : h_keys_temp[i] % FREQ_RANGE;
        if (in_base && in_test) {
          original_score *= 2;
          ASSERT_EQ(h_scores_temp[i],
                    make_expected_score_for_epochlfu<S>(global_epoch,
                                                        original_score));
        } else {
          ASSERT_EQ(h_scores_temp[i],
                    make_expected_score_for_epochlfu<S>(
                        global_epoch - static_cast<S>(in_base),
                        original_score));
        }
      }
    }
    VerifyValuesUsingKeys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
}

void test_evict_strategy_customized_advanced(size_t max_hbm_for_vectors) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 8;
  constexpr uint64_t TEMP_KEY_NUM = BASE_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 256;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY,
                                     max_hbm_for_vectors, 16);
  options.num_of_buckets_per_alloc = 8;
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
      h_vectors_test[i * DIM + j] = static_cast<V>(h_keys_test[i] * 0.00001);
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
    ASSERT_EQ(table->bucket_count(), BUCKET_NUM);
    ASSERT_EQ(table->size(guard.stream), 0);
    d_keys.copy_from_host(h_keys_base, guard.stream);
    d_scores.copy_from_host(h_scores_base, guard.stream);
    d_vectors.copy_from_host(h_vectors_base, guard.stream);
    table->insert_or_assign(BASE_KEY_NUM, d_keys.get(), d_vectors.get(),
                            d_scores.get(), guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream, DIM);
    auto sorted = h_scores_temp;
    std::sort(sorted.begin(), sorted.end());
    auto expected = range<S, TEMP_KEY_NUM>(BASE_SCORE_START);
    ASSERT_TRUE(std::equal(sorted.begin(), sorted.end(), expected.begin()));
    VerifyValuesUsingKeys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    d_keys.copy_from_host(h_keys_test, guard.stream);
    d_scores.copy_from_host(h_scores_test, guard.stream);
    d_vectors.copy_from_host(h_vectors_test, guard.stream);
    table->insert_or_assign(TEST_KEY_NUM, d_keys.get(), d_vectors.get(),
                            d_scores.get(), guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    ASSERT_EQ(table->size(guard.stream), BUCKET_MAX_SIZE);
    export_to_host(table.get(), BUCKET_MAX_SIZE, d_keys.get(), d_vectors.get(),
                 d_scores.get(), &h_keys_temp, &h_vectors_temp, &h_scores_temp,
                 guard.stream, DIM);
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
      if (h_keys_temp[i] == h_keys_test[4]) ASSERT_EQ(h_scores_temp[i], 99);
      if (h_keys_temp[i] == h_keys_test[5]) ASSERT_EQ(h_scores_temp[i], 1010);
      if (h_keys_temp[i] == h_keys_test[6]) ASSERT_EQ(h_scores_temp[i], 1020);
      if (h_keys_temp[i] == h_keys_test[7]) ASSERT_EQ(h_scores_temp[i], 1035);
    }
    VerifyValuesUsingKeys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
}

void test_evict_strategy_customized_correct_rate(size_t max_hbm_for_vectors) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024UL;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr float EXPECTED_CORRECT_RATE = 0.964f;
  constexpr int ROUNDS = 12;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, MAX_CAPACITY,
                                     max_hbm_for_vectors, 17, DIM,
                                     MAX_BUCKET_SIZE);
  options.num_of_buckets_per_alloc = 128;
  StreamGuard guard;
  guard.create();

  std::vector<K> h_keys_base(BATCH_SIZE);
  std::vector<S> h_scores_base(BATCH_SIZE);
  std::vector<V> h_vectors_base(BATCH_SIZE * DIM);
  std::vector<K> h_keys_temp(MAX_CAPACITY);
  std::vector<S> h_scores_temp(MAX_CAPACITY);
  std::vector<V> h_vectors_temp(MAX_CAPACITY * DIM);
  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  d_keys.alloc(MAX_CAPACITY);
  d_scores.alloc(MAX_CAPACITY);
  d_vectors.alloc(MAX_CAPACITY * DIM);

  auto table = std::make_unique<Table>();
  table->init(options);
  size_t start_key = 100000;
  ASSERT_EQ(table->size(guard.stream), 0);
  for (int round = 0; round < ROUNDS; ++round) {
    const size_t expected_min_key = 100000 + INIT_CAPACITY * round;
    const size_t expected_max_key = 100000 + INIT_CAPACITY * (round + 1) - 1;
    const size_t expected_table_size =
        round == 0 ? static_cast<size_t>(EXPECTED_CORRECT_RATE * INIT_CAPACITY)
                   : INIT_CAPACITY;
    for (size_t step = 0; step < STEPS; ++step) {
      create_continuous_keys<K, S, V, DIM>(h_keys_base.data(),
                                           h_scores_base.data(),
                                           h_vectors_base.data(), BATCH_SIZE,
                                           start_key);
      start_key += BATCH_SIZE;
      d_keys.copy_from_host(h_keys_base, guard.stream);
      d_scores.copy_from_host(h_scores_base, guard.stream);
      d_vectors.copy_from_host(h_vectors_base, guard.stream);
      table->insert_or_assign(BATCH_SIZE, d_keys.get(), d_vectors.get(),
                              d_scores.get(), guard.stream);
      ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    }
    const size_t total_size = table->size(guard.stream);
    ASSERT_GE(total_size, expected_table_size);
    ASSERT_EQ(MAX_CAPACITY, table->capacity());
    const size_t dump_counter =
        table->export_batch(MAX_CAPACITY, 0, d_keys.get(), d_vectors.get(),
                            d_scores.get(), guard.stream);
    ASSERT_EQ(dump_counter, total_size);
    copy_device_to_host_raw(d_keys.get(), &h_keys_temp, MAX_CAPACITY, guard.stream);
    copy_device_to_host_raw(d_scores.get(), &h_scores_temp, MAX_CAPACITY,
                        guard.stream);
    copy_device_to_host_raw(d_vectors.get(), &h_vectors_temp, MAX_CAPACITY * DIM,
                        guard.stream);
    size_t bigger_score_counter = 0;
    K max_key = 0;
    for (size_t i = 0; i < dump_counter; ++i) {
      ASSERT_EQ(h_keys_temp[i], h_scores_temp[i]);
      max_key = std::max(max_key, h_keys_temp[i]);
      if (h_scores_temp[i] >= expected_min_key) {
        ++bigger_score_counter;
      }
      for (size_t j = 0; j < DIM; ++j) {
        ASSERT_EQ(h_vectors_temp[i * DIM + j],
                  static_cast<float>(h_keys_temp[i] * 0.00001));
      }
    }
    const float correct_rate =
        (bigger_score_counter * 1.0f) / MAX_CAPACITY;
    std::cout << std::setprecision(3) << "[Round " << round << "]"
              << "correct_rate=" << correct_rate << std::endl;
    ASSERT_GE(max_key, expected_max_key);
    ASSERT_GE(correct_rate, EXPECTED_CORRECT_RATE);
  }
}

void test_insert_or_assign_multi_threads(size_t max_hbm_for_vectors,
                                         const float batch_0_ratio,
                                         const float batch_1_ratio,
                                         bool capacity_silent = true) {
  constexpr uint64_t THREAD_N = 64UL;
  const uint64_t batch_0_size = static_cast<uint64_t>(THREAD_N * batch_0_ratio);
  const uint64_t batch_1_size = static_cast<uint64_t>(THREAD_N * batch_1_ratio);
  constexpr uint64_t INIT_CAPACITY = 32 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = 128 * 1024 * 1024UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  TableOptions options = make_options(INIT_CAPACITY, MAX_CAPACITY,
                                     max_hbm_for_vectors, 0, DIM,
                                     BUCKET_MAX_SIZE);
  options.max_load_factor = 0.50f;
  options.api_lock = true;
  auto table = std::make_shared<Table>();
  table->init(options);

  auto worker = [&table, KEY_NUM, options, capacity_silent](int batch,
                                                            int task_n,
                                                            bool update_twice) {
    auto device_id_env = std::getenv("HKV_TEST_DEVICE");
    int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
    HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                    "aclrtSetDevice failed");
    StreamGuard guard;
    guard.create();
    std::vector<K> h_keys(KEY_NUM);
    std::vector<V> h_vectors(KEY_NUM * options.dim);
    std::vector<uint8_t> h_found(KEY_NUM);
    const size_t current_capacity = table->capacity();
    DeviceArray<K> d_keys;
    DeviceArray<V> d_vectors;
    DeviceArray<V> d_new_vectors;
    DeviceArray<bool> d_found;
    d_keys.alloc(KEY_NUM);
    d_vectors.alloc(KEY_NUM * options.dim);
    d_new_vectors.alloc(KEY_NUM * options.dim);
    d_found.alloc(KEY_NUM);
    create_random_keys<K, S, V, DIM>(h_keys.data(), nullptr, h_vectors.data(),
                                     KEY_NUM);
    d_keys.copy_from_host(h_keys, guard.stream);
    d_vectors.copy_from_host(h_vectors, guard.stream);
    if (!update_twice) {
      table->find(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(),
                  nullptr, guard.stream);
      ACL_CHECK(aclrtSynchronizeStream(guard.stream));
      copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
      ASSERT_EQ(std::count_if(h_found.begin(), h_found.end(),
                              [](uint8_t v) { return v != 0; }),
                0);
    }
    table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(), nullptr,
                            guard.stream);
    if (update_twice) {
      d_new_vectors.memset(2, guard.stream);
      table->insert_or_assign(KEY_NUM, d_keys.get(), d_new_vectors.get(),
                              nullptr, guard.stream);
    }
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    d_vectors.memset(0, guard.stream);
    table->find(KEY_NUM, d_keys.get(), d_vectors.get(), d_found.get(), nullptr,
                guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    copy_bool_to_host(d_found.get(), &h_found, KEY_NUM, guard.stream);
    copy_device_to_host_raw(d_keys.get(), &h_keys, KEY_NUM, guard.stream);
    copy_device_to_host_raw(d_vectors.get(), &h_vectors, KEY_NUM * options.dim,
                        guard.stream);
    size_t found_num = 0;
    size_t err_times = 0;
    const V expected_update = byte_pattern_value<V>();
    for (size_t i = 0; i < KEY_NUM; ++i) {
      if (!h_found[i]) {
        continue;
      }
      ++found_num;
      for (size_t j = 0; j < options.dim; ++j) {
        const V expected =
            update_twice ? expected_update : static_cast<V>(h_keys[i] * 0.00001);
        if (h_vectors[i * options.dim + j] != expected) {
          ++err_times;
          break;
        }
      }
    }
    ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(),
                      found_num, guard.stream);
    if (batch == 0 || batch == 1) {
      ASSERT_EQ(found_num, KEY_NUM);
      ASSERT_EQ(err_times, 0);
    } else if (found_num != KEY_NUM || err_times != 0) {
      std::cout << " [Thread " << task_n << "]\t"
                << "Number of keys(insert/found/error) : (" << KEY_NUM << "/"
                << found_num << "/" << err_times << ") \t" << std::endl;
    }
    if (current_capacity != table->capacity() && !capacity_silent) {
      std::cout << " [Thread " << task_n << "]\t"
                << "The capacity changed from " << current_capacity << " to "
                << table->capacity() << std::endl;
    }
  };

  std::vector<std::thread> threads;
  /* the table is relative idle, and assume there is no eviction */
  int batch = 0;
  std::cout << "[Batch 0] " << batch_0_size << " threads\n";
  for (uint64_t i = 0; i < batch_0_size; i += 2) {
    threads.emplace_back(worker, batch, static_cast<int>(i), false);
    threads.emplace_back(worker, batch, static_cast<int>(i + 1), true);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  threads.clear();

  /* test the correct of APIs serially */
  batch = 1;
  std::cout << "[Batch 1] " << batch_1_size << " threads\n";
  for (uint64_t i = batch_0_size; i < batch_0_size + batch_1_size; i += 2) {
    auto th = std::thread(worker, batch, static_cast<int>(i), false);
    th.join();
    th = std::thread(worker, batch, static_cast<int>(i + 1), true);
    th.join();
  }

  /* eviction may occur */
  batch = 2;
  std::cout << "[Batch 2] "
            << THREAD_N - batch_0_size - batch_1_size << " threads\n";
  for (uint64_t i = batch_0_size + batch_1_size; i < THREAD_N; i += 2) {
    threads.emplace_back(worker, batch, static_cast<int>(i), false);
    threads.emplace_back(worker, batch, static_cast<int>(i + 1), true);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  ASSERT_EQ(table->capacity(), MAX_CAPACITY);
}

template <typename KType, typename VType, typename SType, typename Table,
          size_t dim = 64>
void CheckInsertOrAssignValues(Table* table, KType* keys, VType* values,
                               SType* scores, size_t len,
                               aclrtStream stream) {
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
  copy_device_to_host_raw(d_tmp_keys.get(), &h_tmp_keys, table_size_before,
                      stream);
  copy_device_to_host_raw(d_tmp_values.get(), &h_tmp_values,
                      table_size_before * dim, stream);
  copy_device_to_host_raw(d_tmp_scores.get(), &h_tmp_scores, table_size_before,
                      stream);

  auto start = std::chrono::steady_clock::now();
  table->insert_or_assign(len, keys, values, nullptr, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  auto end = std::chrono::steady_clock::now();
  const auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  const float dur = static_cast<float>(diff.count());
  (void)dur;

  const size_t table_size_after = table->size(stream);
  const size_t table_size_verify1 =
      table->export_batch(table->capacity(), 0, d_tmp_keys.get(),
                          d_tmp_values.get(), d_tmp_scores.get(), stream);
  ASSERT_EQ(table_size_verify1, table_size_after);
  copy_device_to_host_raw(d_tmp_keys.get(), &h_tmp_keys, table_size_after,
                      stream);
  copy_device_to_host_raw(d_tmp_values.get(), &h_tmp_values,
                      table_size_after * dim, stream);
  copy_device_to_host_raw(d_tmp_scores.get(), &h_tmp_scores, table_size_after,
                      stream);

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
  std::cout << "Check insert behavior got value_diff_cnt: " << value_diff_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << std::endl;
}

void test_insert_or_assign_values_check(size_t max_hbm_for_vectors) {
  constexpr size_t U = 524288;
  constexpr size_t INIT_CAPACITY = 1024;
  constexpr size_t B = 524288 + 13;
  constexpr size_t dim = 64;
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  init_env();
  TableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = INIT_CAPACITY;
  opt.max_hbm_for_vectors = GB(max_hbm_for_vectors);
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
    CheckInsertOrAssignValues<K, V, S, Table, dim>(
        table.get(), data_buffer.keys_ptr(), data_buffer.values_ptr(),
        data_buffer.scores_ptr(), B, guard.stream);
    offset += B;
    score += 1;
  }
}

void test_bucket_size(bool load_scores = true) {
  constexpr uint64_t INIT_CAPACITY = 128 * 1024UL;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint32_t TEST_DIM = 4;
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;

  init_env();
  StreamGuard guard;
  guard.create();
  std::vector<K> h_keys(KEY_NUM);
  std::vector<S> h_scores(KEY_NUM);
  std::vector<V> h_vectors(KEY_NUM * TEST_DIM);
  std::vector<uint8_t> h_found(KEY_NUM);
  DeviceArray<K> d_keys;
  DeviceArray<S> d_scores;
  DeviceArray<V> d_vectors;
  DeviceArray<bool> d_found;
  d_keys.alloc(KEY_NUM);
  d_scores.alloc(KEY_NUM);
  d_vectors.alloc(KEY_NUM * TEST_DIM);
  d_found.alloc(KEY_NUM);

  for (uint64_t bucket_max_size = 16; bucket_max_size <= 2048;
       bucket_max_size *= 2) {
    TableOptions options = make_options(INIT_CAPACITY, INIT_CAPACITY, 16, 1,
                                       TEST_DIM, bucket_max_size);
    options.num_of_buckets_per_alloc = 2;
    create_random_keys<K, S, V>(options.dim, h_keys.data(), h_scores.data(),
                                h_vectors.data(), KEY_NUM);
    d_keys.copy_from_host(h_keys, guard.stream);
    d_scores.copy_from_host(h_scores, guard.stream);
    d_vectors.copy_from_host(h_vectors, guard.stream);
    auto table = std::make_unique<Table>();
    table->init(options);
    ASSERT_EQ(table->size(guard.stream), 0);
    table->insert_or_assign(KEY_NUM, d_keys.get(), d_vectors.get(),
                            d_scores.get(), guard.stream);
    ACL_CHECK(aclrtSynchronizeStream(guard.stream));
    ASSERT_EQ(table->size(guard.stream), KEY_NUM);
    d_vectors.memset(0, guard.stream);
    d_scores.memset(0, guard.stream);
    FindAndVerify(table.get(), KEY_NUM, d_keys.get(), d_vectors.get(),
                  d_scores.get(), d_found.get(), &h_keys, &h_scores,
                  &h_vectors, &h_found, guard.stream, TEST_DIM, KEY_NUM,
                  load_scores);
    ContainsAndVerify(table.get(), KEY_NUM, d_keys.get(), d_found.get(),
                      KEY_NUM, guard.stream);
  }
}

}  // namespace

TEST(MerlinHashTableTest, test_export_batch_if) {
  test_export_batch_if(16);
  test_export_batch_if(0);
}

TEST(MerlinHashTableTest, test_insert_or_assign_multi_threads) {
  test_insert_or_assign_multi_threads(16, 0.25f, 0.125f);
  test_insert_or_assign_multi_threads(16, 0.375f, 0.125f);
  test_insert_or_assign_multi_threads(0, 0.25f, 0.125f);
  test_insert_or_assign_multi_threads(0, 0.375f, 0.125f);
}

TEST(MerlinHashTableTest, test_basic) {
  test_basic_common(16, false);
  test_basic_common(0, false);
}

TEST(MerlinHashTableTest, test_basic_without_rehash) {
  test_basic_common(16, true);
  test_basic_common(0, true);
}

TEST(MerlinHashTableTest, test_bucket_size) { test_bucket_size(); }

TEST(MerlinHashTableTest, test_find_using_pipeline) {
  test_find_using_pipeline<int32_t>(224, true);
  test_find_using_pipeline<uint32_t>(202, true);
  test_find_using_pipeline<float>(129, true);

  test_find_using_pipeline<float>(128, true);
  test_find_using_pipeline<int32_t>(66, false);
  test_find_using_pipeline<uint32_t>(3, false);
  test_find_using_pipeline<double>(3, true);

  test_find_using_pipeline<int16_t>(128, true);
  test_find_using_pipeline<int8_t>(66, false);
  test_find_using_pipeline<uint16_t>(3, false);
  test_find_using_pipeline<uint8_t>(3, true);
}

TEST(MerlinHashTableTest, test_basic_when_full) {
  test_basic_when_full(16);
  test_basic_when_full(0);
}

TEST(MerlinHashTableTest, test_erase_if_pred) {
  test_erase_if_pred<EraseIfVersion::V1>(16);
  test_erase_if_pred<EraseIfVersion::V1>(0);
  test_erase_if_pred<EraseIfVersion::V2>(16);
  test_erase_if_pred<EraseIfVersion::V3>(16);
}

TEST(MerlinHashTableTest, test_rehash) {
  test_rehash(16);
  test_rehash(0);
}

TEST(MerlinHashTableTest, test_rehash_on_big_batch_specific) {
  test_rehash_on_big_batch_specific(16);
  test_rehash_on_big_batch_specific(0);
}

TEST(MerlinHashTableTest, test_rehash_on_big_batch) {
  test_rehash_on_big_batch(16);
  test_rehash_on_big_batch(0);
}

TEST(MerlinHashTableTest, test_dynamic_rehash_on_multi_threads) {
  test_dynamic_rehash_on_multi_threads(16);
  test_dynamic_rehash_on_multi_threads(0);
}

TEST(MerlinHashTableTest, test_basic_for_cpu_io) { test_basic_for_cpu_io(); }

TEST(MerlinHashTableTest, test_evict_strategy_lru_basic) {
  test_evict_strategy_basic<EvictStrategy::kLru>(16);
  test_evict_strategy_basic<EvictStrategy::kLru>(0);
}

TEST(MerlinHashTableTest, test_evict_strategy_lfu_basic) {
  test_evict_strategy_basic<EvictStrategy::kLfu>(16);
  // TODO: Add back when diff error issue fixed in hybrid mode.
  // test_evict_strategy_lfu_basic(0);
}

TEST(MerlinHashTableTest, test_evict_strategy_epochlru_basic) {
  test_evict_strategy_basic<EvictStrategy::kEpochLru>(16);
  test_evict_strategy_basic<EvictStrategy::kEpochLru>(0);
}

TEST(MerlinHashTableTest, test_evict_strategy_epochlfu_basic) {
  test_evict_strategy_basic<EvictStrategy::kEpochLfu>(16);
  test_evict_strategy_basic<EvictStrategy::kEpochLfu>(0);
}

TEST(MerlinHashTableTest, test_evict_strategy_customized_basic) {
  test_evict_strategy_basic<EvictStrategy::kCustomized>(16);
  test_evict_strategy_basic<EvictStrategy::kCustomized>(0);
}

TEST(MerlinHashTableTest, test_evict_strategy_customized_advanced) {
  test_evict_strategy_customized_advanced(16);
  test_evict_strategy_customized_advanced(0);
}

TEST(MerlinHashTableTest, test_evict_strategy_customized_correct_rate) {
  test_evict_strategy_customized_correct_rate(16);
  // TODO: after blossom CI issue is resolved, the skip logic.
  const bool skip_hmem_check = (nullptr != std::getenv("IS_BLOSSOM_CI"));
  if (!skip_hmem_check) {
    test_evict_strategy_customized_correct_rate(0);
  } else {
    std::cout << "The HMEM check is skipped in blossom CI!" << std::endl;
  }
}

TEST(MerlinHashTableTest, test_insert_or_assign_values_check) {
  test_insert_or_assign_values_check(16);
  // TODO: Add back when diff error issue fixed in hybrid mode.
  test_insert_or_assign_values_check(0);
}
