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

TEST(AccumOrAssignTest, test_evict_strategy_lru_basic) {
  test_evict_strategy_lru_basic();
}

TEST(AccumOrAssignTest, test_evict_strategy_lfu_basic) {
  test_evict_strategy_lfu_basic();
}
