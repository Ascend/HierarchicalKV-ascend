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
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = HashTableOptions;

template <typename Table>
unique_ptr<Table> make_table(const TableOptions& options) {
  auto table = make_unique<Table>();
  table->init(options);
  return table;
}

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

struct DeviceMem {
  void* ptr = nullptr;
  DeviceMem() = default;
  ~DeviceMem() {
    if (ptr) aclrtFree(ptr);
  }
  DeviceMem(const DeviceMem&) = delete;
  DeviceMem& operator=(const DeviceMem&) = delete;
  DeviceMem(DeviceMem&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }
  DeviceMem& operator=(DeviceMem&& other) noexcept {
    if (this != &other) {
      if (ptr) aclrtFree(ptr);
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }
  template <typename T>
  T* as() { return reinterpret_cast<T*>(ptr); }
  static DeviceMem Alloc(size_t bytes) {
    DeviceMem m;
    EXPECT_EQ(aclrtMalloc(&m.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    return m;
  }
};

void test_evict_strategy_lru_basic() {
  init_env();
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr float true_ratio = 0.5;

  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = GB(16);
  using Table = HashTable<K, V, S, EvictStrategy::kLru>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  auto d_keys = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(V) * DIM);
  auto d_accum_or_assigns = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(bool));

  create_random_bools<K>(
      reinterpret_cast<bool*>(h_accum_or_assigns_base.data()), BASE_KEY_NUM,
      true_ratio);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  create_random_bools<K>(
      reinterpret_cast<bool*>(h_accum_or_assigns_test.data()), TEST_KEY_NUM,
      true_ratio);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  h_accum_or_assigns_test[2] = true;
  h_accum_or_assigns_test[3] = false;

  for (int i = 0; i < DIM; i++) {
    h_vectors_test[2 * DIM + i] = h_vectors_base[72 * DIM + i];
    h_vectors_test[3 * DIM + i] = h_vectors_base[73 * DIM + i];
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    auto table = make_table<Table>(options);

    total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);

    // Phase 1: insert base keys
    {
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            h_keys_base.data(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            h_scores_base.data(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            h_accum_or_assigns_base.data(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      S start_ts = host_nano<S>(stream);
      table->accum_or_assign(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), nullptr, stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      S end_ts = host_nano<S>(stream);

      size_t sz = table->size(stream);
      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      ACL_CHECK(aclrtSynchronizeStream(stream));
      ASSERT_EQ(sz, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);

      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), BASE_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), BASE_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(),
                            BASE_KEY_NUM * sizeof(V) * DIM, d_vectors.as<V>(),
                            BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));

      for (size_t j = 0; j < dump_counter; j++) {
        ASSERT_GE(h_scores_temp[j], start_ts);
        ASSERT_LE(h_scores_temp[j], end_ts);
        for (int k = 0; k < DIM; k++) {
          ASSERT_EQ(h_vectors_temp[j * DIM + k],
                    static_cast<float>(h_keys_temp[j] * 0.00001));
        }
      }
    }

    // Phase 2: insert test keys (with overlap)
    {
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), TEST_KEY_NUM * sizeof(K),
                            h_keys_test.data(), TEST_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), TEST_KEY_NUM * sizeof(S),
                            h_scores_test.data(), TEST_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), TEST_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            TEST_KEY_NUM * sizeof(bool),
                            h_accum_or_assigns_test.data(),
                            TEST_KEY_NUM * sizeof(bool),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      S start_ts = host_nano<S>(stream);
      table->accum_or_assign(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), nullptr, stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      S end_ts = host_nano<S>(stream);

      size_t sz = table->size(stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));

      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      for (int j = 0; j < TEST_KEY_NUM; j++) {
        if ((h_keys_base.end() == find(h_keys_base.begin(), h_keys_base.end(),
                                       h_keys_test[j])) &&
            !h_accum_or_assigns_test[j])
          expected_size++;
      }
      ASSERT_EQ(sz, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);

      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), TEMP_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), TEMP_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), TEMP_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), TEMP_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(),
                            TEMP_KEY_NUM * sizeof(V) * DIM, d_vectors.as<V>(),
                            TEMP_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));

      for (size_t j = 0; j < dump_counter; j++) {
        bool is_accum = (h_keys_temp[j] == h_keys_test[2]);
        bool is_new_insert =
            (h_keys_test.end() !=
             find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[j]));
        if (is_accum) {
          for (int k = 0; k < DIM; k++) {
            ASSERT_EQ(h_vectors_temp[j * DIM + k],
                      static_cast<float>(h_keys_temp[j] * 0.00002));
          }
        } else {
          for (int k = 0; k < DIM; k++) {
            ASSERT_EQ(h_vectors_temp[j * DIM + k],
                      static_cast<float>(h_keys_temp[j] * 0.00001));
          }
        }
        if (is_accum ||
            (is_new_insert && (h_keys_temp[j] != h_keys_test[3]))) {
          ASSERT_GE(h_scores_temp[j], start_ts);
          ASSERT_LE(h_scores_temp[j], end_ts);
        } else {
          ASSERT_LE(h_scores_temp[j], start_ts);
        }
      }
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_lfu_basic() {
  init_env();
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 1024;
  constexpr float true_ratio = 0.5;

  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = GB(16);
  using Table = HashTable<K, V, S, EvictStrategy::kLfu>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);

  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);

  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  auto d_keys = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(V) * DIM);
  auto d_accum_or_assigns = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(bool));

  int freq_range = 1000;

  create_random_bools<K>(
      reinterpret_cast<bool*>(h_accum_or_assigns_base.data()), BASE_KEY_NUM,
      true_ratio);
  create_random_bools<K>(
      reinterpret_cast<bool*>(h_accum_or_assigns_test.data()), TEST_KEY_NUM,
      true_ratio);

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));

  for (int i = 0; i < TEST_TIMES; i++) {
    create_keys_in_one_buckets_lfu<K, S, V, DIM>(
        h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
        BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0,
        0x3FFFFFFFFFFFFFFF, freq_range);
    create_keys_in_one_buckets_lfu<K, S, V, DIM>(
        h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
        TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFD, freq_range);

    h_accum_or_assigns_base[72] = false;
    h_accum_or_assigns_base[73] = false;

    h_keys_test[2] = h_keys_base[72];
    h_keys_test[3] = h_keys_base[73];

    h_accum_or_assigns_test[2] = true;
    h_accum_or_assigns_test[3] = false;

    h_scores_test[2] = h_keys_base[72] % freq_range;
    h_scores_test[3] = h_keys_base[73] % freq_range;

    for (int j = 0; j < DIM; j++) {
      h_vectors_test[2 * DIM + j] = h_vectors_base[72 * DIM + j];
      h_vectors_test[3 * DIM + j] = h_vectors_base[73 * DIM + j];
    }

    size_t dump_counter = 0;
    S global_epoch = 1;
    auto table = make_table<Table>(options);

    size_t total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);

    // Phase 1
    {
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            h_keys_base.data(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            h_scores_base.data(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            h_accum_or_assigns_base.data(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            ACL_MEMCPY_HOST_TO_DEVICE));

      table->set_global_epoch(global_epoch);
      table->accum_or_assign(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), d_scores.as<S>(),
                             stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));

      size_t sz = table->size(stream);
      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      ACL_CHECK(aclrtSynchronizeStream(stream));
      ASSERT_EQ(sz, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);

      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), BASE_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), BASE_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(),
                            BASE_KEY_NUM * sizeof(V) * DIM, d_vectors.as<V>(),
                            BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));

      for (size_t j = 0; j < dump_counter; j++) {
        ASSERT_EQ(h_scores_temp[j], h_keys_temp[j] % freq_range);
        for (int k = 0; k < DIM; k++) {
          ASSERT_EQ(h_vectors_temp[j * DIM + k],
                    static_cast<float>(h_keys_temp[j] * 0.00001));
        }
      }
    }

    // Phase 2
    {
      global_epoch++;
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), TEST_KEY_NUM * sizeof(K),
                            h_keys_test.data(), TEST_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), TEST_KEY_NUM * sizeof(S),
                            h_scores_test.data(), TEST_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), TEST_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            TEST_KEY_NUM * sizeof(bool),
                            h_accum_or_assigns_test.data(),
                            TEST_KEY_NUM * sizeof(bool),
                            ACL_MEMCPY_HOST_TO_DEVICE));

      table->set_global_epoch(global_epoch);
      table->accum_or_assign(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), d_scores.as<S>(),
                             stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));

      size_t sz = table->size(stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      for (int j = 0; j < TEST_KEY_NUM; j++) {
        if ((h_keys_base.end() == find(h_keys_base.begin(), h_keys_base.end(),
                                       h_keys_test[j])) &&
            !h_accum_or_assigns_test[j])
          expected_size++;
      }
      ASSERT_EQ(sz, expected_size);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);

      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), TEMP_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), TEMP_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), TEMP_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), TEMP_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(),
                            TEMP_KEY_NUM * sizeof(V) * DIM, d_vectors.as<V>(),
                            TEMP_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));

      for (size_t j = 0; j < dump_counter; j++) {
        bool is_accum = (h_keys_temp[j] == h_keys_test[2]);
        if (is_accum) {
          ASSERT_EQ(h_scores_temp[j], (h_keys_temp[j] % freq_range) * 2);
          for (int k = 0; k < DIM; k++) {
            ASSERT_EQ(h_vectors_temp[j * DIM + k],
                      static_cast<float>(h_keys_temp[j] * 0.00002));
          }
        } else {
          ASSERT_EQ(h_scores_temp[j], (h_keys_temp[j] % freq_range));
          for (int k = 0; k < DIM; k++) {
            ASSERT_EQ(h_vectors_temp[j * DIM + k],
                      static_cast<float>(h_keys_temp[j] * 0.00001));
          }
        }
      }
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_epochlru_basic() {
  init_env();
  constexpr int RSHIFT_ON_NANO = 20;
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr float true_ratio = 0.5;
  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = GB(16);
  using Table = HashTable<K, V, S, EvictStrategy::kEpochLru>;
  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  auto d_keys = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(V) * DIM);
  auto d_accum_or_assigns = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(bool));
  create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                         BASE_KEY_NUM, true_ratio);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);
  create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                         TEST_KEY_NUM, true_ratio);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);
  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  h_accum_or_assigns_test[2] = true;
  h_accum_or_assigns_test[3] = false;
  for (int j = 0; j < DIM; j++) {
    h_vectors_test[2 * DIM + j] = h_vectors_base[72 * DIM + j];
    h_vectors_test[3 * DIM + j] = h_vectors_base[73 * DIM + j];
  }
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  size_t dump_counter = 0;
  S global_epoch = 1;
  for (int i = 0; i < TEST_TIMES; i++) {
    auto table = make_table<Table>(options);
    size_t total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);
    // Phase 1
    {
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            h_keys_base.data(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            h_scores_base.data(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_base.data(), BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            h_accum_or_assigns_base.data(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      S start_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      table->set_global_epoch(global_epoch);
      table->accum_or_assign(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), nullptr, stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      S end_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      size_t sz = table->size(stream);
      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      ACL_CHECK(aclrtSynchronizeStream(stream));
      ASSERT_EQ(sz, expected_size);
      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);
      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), BASE_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), BASE_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(), BASE_KEY_NUM * sizeof(V) * DIM,
                            d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));
      for (size_t j = 0; j < dump_counter; j++) {
        ASSERT_GE(h_scores_temp[j] & 0xFFFFFFFF, start_ts);
        ASSERT_LE(h_scores_temp[j] & 0xFFFFFFFF, end_ts);
        for (int k = 0; k < DIM; k++) {
          ASSERT_EQ(h_vectors_temp[j * DIM + k],
                    static_cast<float>(h_keys_temp[j] * 0.00001));
        }
      }
    }
    // Phase 2
    {
      global_epoch++;
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), TEST_KEY_NUM * sizeof(K),
                            h_keys_test.data(), TEST_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), TEST_KEY_NUM * sizeof(S),
                            h_scores_test.data(), TEST_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), TEST_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_test.data(), TEST_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            TEST_KEY_NUM * sizeof(bool),
                            h_accum_or_assigns_test.data(),
                            TEST_KEY_NUM * sizeof(bool),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      S start_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      table->set_global_epoch(global_epoch);
      table->accum_or_assign(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), nullptr, stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      S end_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
      size_t sz = table->size(stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      for (int j = 0; j < TEST_KEY_NUM; j++) {
        if ((h_keys_base.end() ==
             find(h_keys_base.begin(), h_keys_base.end(), h_keys_test[j])) &&
            !h_accum_or_assigns_test[j])
          expected_size++;
      }
      ASSERT_EQ(sz, expected_size);
      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);
      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), TEMP_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), TEMP_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), TEMP_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), TEMP_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(), TEMP_KEY_NUM * sizeof(V) * DIM,
                            d_vectors.as<V>(), TEMP_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));
      for (size_t j = 0; j < dump_counter; j++) {
        bool is_accum = (h_keys_temp[j] == h_keys_test[2]);
        bool is_new_insert = (h_keys_test.end() !=
                              find(h_keys_test.begin(), h_keys_test.end(),
                                   h_keys_temp[j]));
        if (is_accum) {
          for (int k = 0; k < DIM; k++) {
            ASSERT_EQ(h_vectors_temp[j * DIM + k],
                      static_cast<float>(h_keys_temp[j] * 0.00002));
          }
        } else {
          for (int k = 0; k < DIM; k++) {
            ASSERT_EQ(h_vectors_temp[j * DIM + k],
                      static_cast<float>(h_keys_temp[j] * 0.00001));
          }
        }
        if (is_accum || (is_new_insert && (h_keys_temp[j] != h_keys_test[3]))) {
          ASSERT_GE(h_scores_temp[j] & 0xffffffff, start_ts);
          ASSERT_LE(h_scores_temp[j] & 0xffffffff, end_ts);
          ASSERT_EQ((h_scores_temp[j] >> 32) & 0xffffffff, global_epoch);
        } else {
          ASSERT_LE(h_scores_temp[j] & 0xffffffff, start_ts);
          ASSERT_EQ((h_scores_temp[j] >> 32) & 0xffffffff, global_epoch - 1);
        }
      }
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_epochlfu_basic() {
  init_env();
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 1024;
  constexpr float true_ratio = 0.5;
  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = GB(16);
  using Table = HashTable<K, V, S, EvictStrategy::kEpochLfu>;
  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  auto d_keys = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(V) * DIM);
  auto d_accum_or_assigns = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(bool));
  int freq_range = 1000;
  create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                         BASE_KEY_NUM, true_ratio);
  create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                         TEST_KEY_NUM, true_ratio);
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
      freq_range);
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD, freq_range);
  h_accum_or_assigns_base[71] = false;
  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;
  h_scores_base[71] = static_cast<S>(numeric_limits<uint32_t>::max() -
                                     static_cast<uint32_t>(1));
  h_keys_test[1] = h_keys_base[71];
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  h_accum_or_assigns_test[1] = true;
  h_accum_or_assigns_test[2] = true;
  h_accum_or_assigns_test[3] = false;
  h_scores_test[1] = h_scores_base[71];
  h_scores_test[2] = h_keys_base[72] % freq_range;
  h_scores_test[3] = h_keys_base[73] % freq_range;
  for (int j = 0; j < DIM; j++) {
    h_vectors_test[1 * DIM + j] = h_vectors_base[71 * DIM + j];
    h_vectors_test[2 * DIM + j] = h_vectors_base[72 * DIM + j];
    h_vectors_test[3 * DIM + j] = h_vectors_base[73 * DIM + j];
  }
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  size_t dump_counter = 0;
  S global_epoch = 1;
  for (int i = 0; i < TEST_TIMES; i++) {
    auto table = make_table<Table>(options);
    size_t total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);
    // Phase 1
    {
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            h_keys_base.data(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            h_scores_base.data(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_base.data(), BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            h_accum_or_assigns_base.data(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      table->set_global_epoch(global_epoch);
      table->accum_or_assign(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), d_scores.as<S>(),
                             stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t sz = table->size(stream);
      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      ACL_CHECK(aclrtSynchronizeStream(stream));
      ASSERT_EQ(sz, expected_size);
      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);
      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), BASE_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), BASE_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(), BASE_KEY_NUM * sizeof(V) * DIM,
                            d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));
      for (size_t j = 0; j < dump_counter; j++) {
        if (h_keys_temp[j] == h_keys_base[71]) {
          S expected_score = make_expected_score_for_epochlfu<S>(
              global_epoch, h_scores_base[71]);
          ASSERT_EQ(h_scores_temp[j], expected_score);
        } else {
          S expected_score = make_expected_score_for_epochlfu<S>(
              global_epoch, (h_keys_temp[j] % freq_range));
          ASSERT_EQ(h_scores_temp[j], expected_score);
        }
        for (int k = 0; k < DIM; k++) {
          ASSERT_EQ(h_vectors_temp[j * DIM + k],
                    static_cast<float>(h_keys_temp[j] * 0.00001));
        }
      }
    }
    // Phase 2
    {
      global_epoch++;
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), TEST_KEY_NUM * sizeof(K),
                            h_keys_test.data(), TEST_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), TEST_KEY_NUM * sizeof(S),
                            h_scores_test.data(), TEST_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), TEST_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_test.data(), TEST_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            TEST_KEY_NUM * sizeof(bool),
                            h_accum_or_assigns_test.data(),
                            TEST_KEY_NUM * sizeof(bool),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      table->set_global_epoch(global_epoch);
      table->accum_or_assign(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), d_scores.as<S>(),
                             stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t sz = table->size(stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t expected_size = 0;
      for (int j = 0; j < BASE_KEY_NUM; j++) {
        if (!h_accum_or_assigns_base[j]) expected_size++;
      }
      for (int j = 0; j < TEST_KEY_NUM; j++) {
        if ((h_keys_base.end() ==
             find(h_keys_base.begin(), h_keys_base.end(), h_keys_test[j])) &&
            !h_accum_or_assigns_test[j])
          expected_size++;
      }
      ASSERT_EQ(sz, expected_size);
      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);
      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), TEMP_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), TEMP_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), TEMP_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), TEMP_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(), TEMP_KEY_NUM * sizeof(V) * DIM,
                            d_vectors.as<V>(), TEMP_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ASSERT_TRUE(h_keys_temp.end() !=
                  find(h_keys_temp.begin(), h_keys_temp.end(), h_keys_base[71]));
      for (size_t j = 0; j < dump_counter; j++) {
        bool in_base = h_keys_base.end() !=
                       find(h_keys_base.begin(), h_keys_base.end(),
                            h_keys_temp[j]);
        bool is_accum = (h_keys_temp[j] == h_keys_test[1] ||
                         h_keys_temp[j] == h_keys_test[2]);
        bool is_new_insert =
            (h_keys_test.end() !=
             find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[j]));
        if (is_accum) {
          if (h_keys_temp[j] == h_keys_base[71]) {
            S expected_score = make_expected_score_for_epochlfu<S>(
                global_epoch, h_scores_base[71] * 2);
            ASSERT_EQ(h_scores_temp[j], expected_score);
          } else {
            S expected_score = make_expected_score_for_epochlfu<S>(
                global_epoch, (h_keys_temp[j] % freq_range) * 2);
            ASSERT_EQ(h_scores_temp[j], expected_score);
          }
        } else {
          if (h_keys_temp[j] == h_keys_base[71]) {
            S expected_score = make_expected_score_for_epochlfu<S>(
                global_epoch, h_scores_base[71] * 2);
            ASSERT_EQ(h_scores_temp[j], expected_score);
          } else {
            S expected_score = make_expected_score_for_epochlfu<S>(
                global_epoch - static_cast<S>(in_base),
                (h_keys_temp[j] % freq_range));
            ASSERT_EQ(h_scores_temp[j], expected_score);
          }
        }
        for (int k = 0; k < DIM; k++) {
          ASSERT_EQ(h_vectors_temp[j * DIM + k],
                    static_cast<float>(h_keys_temp[j] *
                                       (is_accum ? 0.00002 : 0.00001)))
              << ",j=" << j << ",is_accum=" << is_accum;
        }
      }
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_customized_advanced() {
  init_env();
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 8;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 256;
  constexpr float base_true_ratio = 0.0f;
  constexpr float test_true_ratio = 0.5f;
  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = GB(16);
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;
  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_base(BASE_KEY_NUM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns_test(TEST_KEY_NUM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  auto d_keys = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(V) * DIM);
  auto d_accum_or_assigns = DeviceMem::Alloc(TEMP_KEY_NUM * sizeof(bool));
  create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
                         BASE_KEY_NUM, base_true_ratio);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);
  const S base_score_start = 1000;
  for (int j = 0; j < BASE_KEY_NUM; j++) {
    h_scores_base[j] = base_score_start + j;
  }
  create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns_test.data()),
                         TEST_KEY_NUM, test_true_ratio);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);
  h_keys_test[4] = h_keys_base[72];
  h_keys_test[5] = h_keys_base[73];
  h_keys_test[6] = h_keys_base[74];
  h_keys_test[7] = h_keys_base[75];
  h_accum_or_assigns_base[72] = false;
  h_accum_or_assigns_base[73] = false;
  h_accum_or_assigns_base[74] = false;
  h_accum_or_assigns_base[75] = false;
  h_scores_test[0] = 20;
  h_scores_test[1] = 78;
  h_scores_test[2] = 97;
  h_scores_test[3] = 98;
  h_scores_test[4] = 99;
  h_scores_test[5] = 1010;
  h_scores_test[6] = 1020;
  h_scores_test[7] = 1035;
  h_accum_or_assigns_test[0] = false;
  h_accum_or_assigns_test[1] = false;
  h_accum_or_assigns_test[2] = false;
  h_accum_or_assigns_test[3] = false;
  h_accum_or_assigns_test[4] = true;
  h_accum_or_assigns_test[5] = true;
  h_accum_or_assigns_test[6] = true;
  h_accum_or_assigns_test[7] = false;
  for (int j = 4; j < TEST_KEY_NUM; j++) {
    for (int k = 0; k < DIM; k++) {
      h_vectors_test[j * DIM + k] = static_cast<V>(h_keys_test[j] * 0.00001);
    }
  }
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    auto table = make_table<Table>(options);
    size_t total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);
    // Phase 1: insert base
    {
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            h_keys_base.data(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            h_scores_base.data(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_base.data(), BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            h_accum_or_assigns_base.data(),
                            BASE_KEY_NUM * sizeof(uint8_t),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      table->accum_or_assign(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), d_scores.as<S>(),
                             stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t sz = table->size(stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t expected_size = 0;
      for (const auto accum : h_accum_or_assigns_base) {
        if (!accum) expected_size++;
      }
      ASSERT_EQ(sz, expected_size);
      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);
      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), BASE_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), BASE_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), BASE_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), BASE_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(), BASE_KEY_NUM * sizeof(V) * DIM,
                            d_vectors.as<V>(), BASE_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));
      vector<S> h_scores_temp_sorted(h_scores_temp);
      sort(h_scores_temp_sorted.begin(), h_scores_temp_sorted.end());
      for (size_t j = 0; j < dump_counter; j++) {
        S expected_score = 0ul;
        for (int k = 0; k < BASE_KEY_NUM; k++) {
          if (h_keys_temp[j] == h_keys_base[k]) {
            expected_score = h_scores_base[k];
            break;
          }
        }
        ASSERT_EQ(h_scores_temp[j], expected_score);
        for (int k = 0; k < DIM; k++) {
          ASSERT_EQ(h_vectors_temp[j * DIM + k],
                    static_cast<float>(h_keys_temp[j] * 0.00001));
        }
      }
    }
    // Phase 2: insert test keys with accum/assign
    {
      ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), TEST_KEY_NUM * sizeof(K),
                            h_keys_test.data(), TEST_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), TEST_KEY_NUM * sizeof(S),
                            h_scores_test.data(), TEST_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                            TEST_KEY_NUM * sizeof(bool),
                            h_accum_or_assigns_test.data(),
                            TEST_KEY_NUM * sizeof(bool),
                            ACL_MEMCPY_HOST_TO_DEVICE));
      ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), TEST_KEY_NUM * sizeof(V) * DIM,
                            h_vectors_test.data(), TEST_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_HOST_TO_DEVICE));
      table->accum_or_assign(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                             d_accum_or_assigns.as<bool>(), d_scores.as<S>(),
                             stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t sz = table->size(stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t expected_size = 0;
      for (const auto accum : h_accum_or_assigns_base) {
        if (!accum) expected_size++;
      }
      expected_size = max(expected_size, static_cast<size_t>(BUCKET_MAX_SIZE));
      ASSERT_EQ(sz, expected_size);
      dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                         d_vectors.as<V>(), d_scores.as<S>(),
                                         stream);
      ASSERT_EQ(dump_counter, expected_size);
      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), TEMP_KEY_NUM * sizeof(K),
                            d_keys.as<K>(), TEMP_KEY_NUM * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), TEMP_KEY_NUM * sizeof(S),
                            d_scores.as<S>(), TEMP_KEY_NUM * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(), TEMP_KEY_NUM * sizeof(V) * DIM,
                            d_vectors.as<V>(), TEMP_KEY_NUM * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));
      for (int j = 0; j < TEST_KEY_NUM; j++) {
        if (j < 4) {
          ASSERT_EQ(h_keys_temp.end(),
                    find(h_keys_temp.begin(), h_keys_temp.end(),
                         h_keys_test[j]));
        } else {
          ASSERT_NE(h_keys_temp.end(),
                    find(h_keys_temp.begin(), h_keys_temp.end(),
                         h_keys_test[j]));
        }
      }
      for (size_t j = 0; j < TEMP_KEY_NUM; j++) {
        if (h_keys_temp[j] == h_keys_test[4])
          ASSERT_EQ(h_scores_temp[j], h_scores_test[4]);
        if (h_keys_temp[j] == h_keys_test[5])
          ASSERT_EQ(h_scores_temp[j], h_scores_test[5]);
        if (h_keys_temp[j] == h_keys_test[6])
          ASSERT_EQ(h_scores_temp[j], h_scores_test[6]);
        if (h_keys_temp[j] == h_keys_test[7])
          ASSERT_NE(h_scores_temp[j], h_scores_test[7]);
        bool is_accum =
            (h_keys_temp[j] != h_keys_test[7]) &&
            (h_keys_test.end() != find(h_keys_test.begin() + 4,
                                       h_keys_test.end(), h_keys_temp[j]));
        for (int k = 0; k < DIM; k++) {
          ASSERT_EQ(h_vectors_temp[j * DIM + k],
                    static_cast<float>(h_keys_temp[j] *
                                       (is_accum ? 0.00002 : 0.00001)));
        }
      }
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_customized_correct_rate() {
  init_env();
  constexpr uint64_t BATCH_SIZE = 1024 * 1024ul;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;
  float expected_correct_rate = 0.964;
  const int rounds = 3;
  constexpr float true_ratio = 0.0;
  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = MAX_BUCKET_SIZE;
  options.max_hbm_for_vectors = GB(16);
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;
  vector<K> h_keys_base(BATCH_SIZE);
  vector<S> h_scores_base(BATCH_SIZE);
  vector<V> h_vectors_base(BATCH_SIZE * DIM);
  vector<uint8_t> h_accum_or_assigns_base(BATCH_SIZE);
  vector<K> h_keys_temp(MAX_CAPACITY);
  vector<S> h_scores_temp(MAX_CAPACITY);
  vector<V> h_vectors_temp(MAX_CAPACITY * DIM);
  auto d_keys = DeviceMem::Alloc(MAX_CAPACITY * sizeof(K));
  auto d_scores = DeviceMem::Alloc(MAX_CAPACITY * sizeof(S));
  auto d_vectors = DeviceMem::Alloc(MAX_CAPACITY * sizeof(V) * DIM);
  auto d_accum_or_assigns = DeviceMem::Alloc(MAX_CAPACITY * sizeof(bool));
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  size_t global_start_key = 100000;
  for (int i = 0; i < TEST_TIMES; i++) {
    auto table = make_table<Table>(options);
    size_t start_key = global_start_key;
    size_t total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);
    for (int r = 0; r < rounds; r++) {
      size_t expected_min_key = global_start_key + INIT_CAPACITY * r;
      size_t expected_max_key = global_start_key + INIT_CAPACITY * (r + 1) - 1;
      size_t expected_table_size =
          (r == 0) ? size_t(expected_correct_rate * INIT_CAPACITY)
                   : INIT_CAPACITY;
      for (int s = 0; s < STEPS; s++) {
        create_random_bools<K>(
            reinterpret_cast<bool*>(h_accum_or_assigns_base.data()),
            BATCH_SIZE, true_ratio);
        create_continuous_keys<K, S, V, DIM>(
            h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
            BATCH_SIZE, start_key);
        start_key += BATCH_SIZE;
        ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), BATCH_SIZE * sizeof(K),
                              h_keys_base.data(), BATCH_SIZE * sizeof(K),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), BATCH_SIZE * sizeof(S),
                              h_scores_base.data(), BATCH_SIZE * sizeof(S),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), BATCH_SIZE * sizeof(V) * DIM,
                              h_vectors_base.data(), BATCH_SIZE * sizeof(V) * DIM,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                              BATCH_SIZE * sizeof(uint8_t),
                              h_accum_or_assigns_base.data(),
                              BATCH_SIZE * sizeof(uint8_t),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        table->accum_or_assign(BATCH_SIZE, d_keys.as<K>(), d_vectors.as<V>(),
                               d_accum_or_assigns.as<bool>(),
                               d_scores.as<S>(), stream);
        ACL_CHECK(aclrtSynchronizeStream(stream));
      }
      total_size = table->size(stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      ASSERT_GE(total_size, expected_table_size);
      ASSERT_EQ(MAX_CAPACITY, table->capacity());
      size_t dump_counter =
          table->export_batch(MAX_CAPACITY, 0, d_keys.as<K>(),
                              d_vectors.as<V>(), d_scores.as<S>(), stream);
      ACL_CHECK(aclrtMemcpy(h_keys_temp.data(), MAX_CAPACITY * sizeof(K),
                            d_keys.as<K>(), MAX_CAPACITY * sizeof(K),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_scores_temp.data(), MAX_CAPACITY * sizeof(S),
                            d_scores.as<S>(), MAX_CAPACITY * sizeof(S),
                            ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK(aclrtMemcpy(h_vectors_temp.data(), MAX_CAPACITY * sizeof(V) * DIM,
                            d_vectors.as<V>(), MAX_CAPACITY * sizeof(V) * DIM,
                            ACL_MEMCPY_DEVICE_TO_HOST));
      size_t bigger_score_counter = 0;
      K max_key = 0;
      size_t values_error_counter = 0;
      for (size_t j = 0; j < dump_counter; j++) {
        ASSERT_EQ(h_keys_temp[j], h_scores_temp[j]);
        max_key = max(max_key, h_keys_temp[j]);
        if (h_scores_temp[j] >= expected_min_key) bigger_score_counter++;
        for (int k = 0; k < DIM; k++) {
          if (h_vectors_temp[j * DIM + k] !=
              static_cast<float>(h_keys_temp[j] * 0.00001)) {
            values_error_counter++;
          }
        }
      }
      ASSERT_EQ(values_error_counter, 0);
      float correct_rate = (bigger_score_counter * 1.0) / MAX_CAPACITY;
      cout << setprecision(3) << "[Round " << r << "]"
           << "correct_rate=" << correct_rate << endl;
      ASSERT_GE(max_key, expected_max_key);
      ASSERT_GE(correct_rate, expected_correct_rate);
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_rehash() {
  init_env();
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
  constexpr uint64_t INIT_CAPACITY = BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = 4 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = BUCKET_MAX_SIZE * 2;
  constexpr uint64_t TEST_TIMES = 100;
  constexpr float true_ratio = 0.5;
  TableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = GB(16);
  using Table = HashTable<K, V, S, EvictStrategy::kCustomized>;
  vector<K> h_keys(KEY_NUM);
  vector<S> h_scores(KEY_NUM);
  vector<V> h_vectors(KEY_NUM * DIM);
  vector<uint8_t> h_accum_or_assigns(KEY_NUM);
  auto d_keys = DeviceMem::Alloc(KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::Alloc(KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::Alloc(KEY_NUM * sizeof(V) * DIM);
  auto d_accum_or_assigns = DeviceMem::Alloc(KEY_NUM * sizeof(bool));
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  for (int i = 0; i < TEST_TIMES; i++) {
    auto table = make_table<Table>(options);
    create_random_bools<K>(reinterpret_cast<bool*>(h_accum_or_assigns.data()),
                           KEY_NUM, true_ratio);
    create_keys_in_one_buckets<K, S, V, DIM>(
        h_keys.data(), h_scores.data(), h_vectors.data(), KEY_NUM,
        INIT_CAPACITY, BUCKET_MAX_SIZE);
    ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), KEY_NUM * sizeof(K), h_keys.data(),
                          KEY_NUM * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_scores.as<S>(), KEY_NUM * sizeof(S),
                          h_scores.data(), KEY_NUM * sizeof(S),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_vectors.as<V>(), KEY_NUM * sizeof(V) * DIM,
                          h_vectors.data(), KEY_NUM * sizeof(V) * DIM,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(d_accum_or_assigns.as<bool>(),
                          KEY_NUM * sizeof(uint8_t), h_accum_or_assigns.data(),
                          KEY_NUM * sizeof(uint8_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    size_t total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);
    table->accum_or_assign(KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                           d_accum_or_assigns.as<bool>(), d_scores.as<S>(),
                           stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t expected_size = 0;
    for (size_t j = 0; j < KEY_NUM; j++) {
      if (!h_accum_or_assigns[j]) expected_size++;
    }
    ASSERT_EQ(total_size, expected_size);
    size_t dump_counter = table->export_batch(
        table->capacity(), 0, d_keys.as<K>(), d_vectors.as<V>(),
        d_scores.as<S>(), stream);
    ASSERT_EQ(dump_counter, expected_size);
    table->reserve(MAX_CAPACITY, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->capacity(), MAX_CAPACITY);
    total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, expected_size);
    dump_counter = table->export_batch(table->capacity(), 0, d_keys.as<K>(),
                                       d_vectors.as<V>(), d_scores.as<S>(),
                                       stream);
    ASSERT_EQ(dump_counter, expected_size);
    ACL_CHECK(aclrtMemcpy(h_keys.data(), KEY_NUM * sizeof(K), d_keys.as<K>(),
                          KEY_NUM * sizeof(K), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(h_scores.data(), KEY_NUM * sizeof(S),
                          d_scores.as<S>(), KEY_NUM * sizeof(S),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy(h_vectors.data(), KEY_NUM * sizeof(V) * DIM,
                          d_vectors.as<V>(), KEY_NUM * sizeof(V) * DIM,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t j = 0; j < dump_counter; j++) {
      ASSERT_EQ(h_scores[j], h_keys[j]);
      for (int k = 0; k < DIM; k++) {
        ASSERT_EQ(h_vectors[j * DIM + k],
                  static_cast<float>(h_keys[j] * 0.00001));
      }
    }
    table->clear(stream);
    total_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

// hybrid模式（HBM+HMEM）下 accum_or_assign SIMD value-copy 测试
// capacity=128, max_hbm_for_vectors 仅容纳一半 value，
// 插入 key_num=1024 >> capacity，确保表满且部分 value 落在 host 内存。
// Phase 1: 验证 ASSIGN — 全部 accum_or_assigns=false
// Phase 2: 验证 ACCUM  — 全部 accum_or_assigns=true，值应变为 2 倍
static void test_accum_or_assign_hybrid_dim(size_t dim) {
  init_env();
  SCOPED_TRACE(::testing::Message() << "dim = " << dim);

  constexpr size_t capacity = 128UL;
  constexpr size_t key_num = 100;

  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = capacity * sizeof(V) * dim / 2;
  options.dim = dim;

  using Table = HashTable<K, V, S>;
  auto table = make_table<Table>(options);
  ASSERT_EQ(table->size(), 0);

  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  create_continuous_keys<K, S, V>(dim, host_keys.data(),
                                  static_cast<S*>(nullptr),
                                  host_values.data(), key_num, 1);

  auto d_keys = DeviceMem::Alloc(key_num * sizeof(K));
  auto d_values = DeviceMem::Alloc(key_num * sizeof(V) * dim);
  auto d_accum = DeviceMem::Alloc(key_num * sizeof(bool));
  auto d_found = DeviceMem::Alloc(key_num * sizeof(bool));

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMemcpy(d_keys.as<K>(), key_num * sizeof(K),
                        host_keys.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // ---- Phase 1: ASSIGN (accum_or_assigns 全 false) ----
  ACL_CHECK(aclrtMemcpy(d_values.as<V>(), key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemset(d_accum.ptr, key_num * sizeof(bool), 0,
                        key_num * sizeof(bool)));

  table->accum_or_assign(key_num, d_keys.as<K>(), d_values.as<V>(),
                         d_accum.as<bool>(), nullptr, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  size_t table_size = table->size(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  EXPECT_EQ(table_size, key_num);

  // 验证 ASSIGN: find 回读 value 应与原始一致
  ACL_CHECK(aclrtMemset(d_values.ptr, key_num * sizeof(V) * dim, 0,
                        key_num * sizeof(V) * dim));

  table->find(key_num, d_keys.as<K>(), d_values.as<V>(),
              d_found.as<bool>(), nullptr, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  bool* host_found = nullptr;
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)));
  ACL_CHECK(aclrtMemcpy(host_found, key_num * sizeof(bool),
                        d_found.as<bool>(), key_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST));

  vector<V> result_values(key_num * dim);
  ACL_CHECK(aclrtMemcpy(result_values.data(), key_num * dim * sizeof(V),
                        d_values.as<V>(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST));

  size_t found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_EQ(result_values[i * dim + j], host_values[i * dim + j])
            << "ASSIGN: key=" << host_keys[i] << " dim_idx=" << j;
      }
    }
  }
  EXPECT_EQ(found_num, key_num);

  // ---- Phase 2: ACCUM (accum_or_assigns 全 true) ----
  ACL_CHECK(aclrtMemcpy(d_values.as<V>(), key_num * sizeof(V) * dim,
                        host_values.data(), key_num * sizeof(V) * dim,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemset(d_accum.ptr, key_num * sizeof(bool), 1,
                        key_num * sizeof(bool)));

  table->accum_or_assign(key_num, d_keys.as<K>(), d_values.as<V>(),
                         d_accum.as<bool>(), nullptr, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  table_size = table->size(stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  EXPECT_EQ(table_size, key_num);

  // 验证 ACCUM: find 回读 value 应为原始值的 2 倍
  ACL_CHECK(aclrtMemset(d_values.ptr, key_num * sizeof(V) * dim, 0,
                        key_num * sizeof(V) * dim));

  table->find(key_num, d_keys.as<K>(), d_values.as<V>(),
              d_found.as<bool>(), nullptr, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(result_values.data(), key_num * dim * sizeof(V),
                        d_values.as<V>(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(host_found, key_num * sizeof(bool),
                        d_found.as<bool>(), key_num * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST));

  found_num = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      found_num++;
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(result_values[i * dim + j],
                        host_values[i * dim + j] * 2)
            << "ACCUM: key=" << host_keys[i] << " dim_idx=" << j;
      }
    }
  }
  EXPECT_EQ(found_num, key_num);

  ACL_CHECK(aclrtFreeHost(host_found));
  ACL_CHECK(aclrtDestroyStream(stream));
}

TEST(AccumOrAssignTest, HybridMode_Dim8) {
  test_accum_or_assign_hybrid_dim(8);
}
TEST(AccumOrAssignTest, HybridMode_Dim1024) {
  test_accum_or_assign_hybrid_dim(1024);
}

TEST(AccumOrAssignTest, test_evict_strategy_lru_basic) {
  test_evict_strategy_lru_basic();
}
TEST(AccumOrAssignTest, test_evict_strategy_lfu_basic) {
  test_evict_strategy_lfu_basic();
}
TEST(AccumOrAssignTest, test_evict_strategy_epochlru_basic) {
  test_evict_strategy_epochlru_basic();
}
TEST(AccumOrAssignTest, test_evict_strategy_epochlfu_basic) {
  test_evict_strategy_epochlfu_basic();
}
TEST(AccumOrAssignTest, test_evict_strategy_customized_advanced) {
  test_evict_strategy_customized_advanced();
}
// run case need add --gtest_also_run_disabled_tests
TEST(AccumOrAssignTest, DISABLED_test_evict_strategy_customized_correct_rate) {
  test_evict_strategy_customized_correct_rate();
}
TEST(AccumOrAssignTest, test_rehash) {
  test_rehash();
}
