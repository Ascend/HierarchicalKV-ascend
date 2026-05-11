/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace community_test_util;

namespace {

constexpr size_t DIM = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = HashTableOptions;

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

struct DeviceMem {
  void* ptr = nullptr;

  DeviceMem() = default;
  ~DeviceMem() {
    if (ptr != nullptr) {
      aclrtFree(ptr);
    }
  }

  DeviceMem(const DeviceMem&) = delete;
  DeviceMem& operator=(const DeviceMem&) = delete;

  DeviceMem(DeviceMem&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
  }

  DeviceMem& operator=(DeviceMem&& other) noexcept {
    if (this != &other) {
      if (ptr != nullptr) {
        aclrtFree(ptr);
      }
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  template <typename T>
  T* as() {
    return reinterpret_cast<T*>(ptr);
  }

  static DeviceMem alloc(size_t bytes) {
    DeviceMem mem;
    EXPECT_EQ(aclrtMalloc(&mem.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    return mem;
  }
};

template <typename T>
void copy_to_device(DeviceMem& dst, const vector<T>& src, size_t count) {
  ACL_CHECK(aclrtMemcpy(dst.as<T>(), count * sizeof(T), src.data(),
                        count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
void copy_to_host(vector<T>& dst, DeviceMem& src, size_t count) {
  ACL_CHECK(aclrtMemcpy(dst.data(), count * sizeof(T), src.as<T>(),
                        count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));
}

template <typename Table>
unique_ptr<Table> make_table(const TableOptions& options) {
  auto table = make_unique<Table>();
  table->init(options);
  return table;
}

TableOptions make_options(size_t max_hbm_for_vectors, int key_start,
                          size_t init_capacity, size_t max_capacity,
                          size_t dim, size_t max_bucket_size = 128) {
  TableOptions options;
  options.reserved_key_start_bit = key_start;
  options.init_capacity = init_capacity;
  options.max_capacity = max_capacity;
  options.max_hbm_for_vectors = GB(max_hbm_for_vectors);
  options.max_bucket_size = max_bucket_size;
  options.dim = dim;
  return options;
}

template <typename T, size_t N>
array<T, N> range_from(const T start) {
  array<T, N> ret{};
  for (size_t i = 0; i < N; ++i) {
    ret[i] = start + static_cast<T>(i);
  }
  return ret;
}

void verify_values_match_keys(const vector<K>& keys, const vector<V>& values,
                              size_t count, size_t dim) {
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(values[i * dim + j],
                static_cast<float>(keys[i] * 0.00001));
    }
  }
}

template <typename Table>
void export_table(Table* table, size_t expected_count, DeviceMem& keys,
                  DeviceMem& values, DeviceMem& scores,
                  vector<K>& host_keys, vector<V>& host_values,
                  vector<S>& host_scores, aclrtStream stream) {
  const size_t exported =
      table->export_batch(table->capacity(), 0, keys.as<K>(), values.as<V>(),
                          scores.as<S>(), stream);
  ASSERT_EQ(exported, expected_count);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  copy_to_host(host_keys, keys, expected_count);
  copy_to_host(host_scores, scores, expected_count);
  copy_to_host(host_values, values, expected_count * DIM);
}

template <int Strategy>
using TableT = HashTable<K, V, S, Strategy>;

void test_evict_strategy_lru_basic(size_t max_hbm_for_vectors,
                                   int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;

  auto options =
      make_options(max_hbm_for_vectors, key_start, INIT_CAPACITY, MAX_CAPACITY,
                   DIM, BUCKET_MAX_SIZE);
  using Table = TableT<EvictStrategy::kLru>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  auto d_keys = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(V) * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  for (int i = 0; i < static_cast<int>(DIM); ++i) {
    h_vectors_test[2 * DIM + i] = h_vectors_base[72 * DIM + i];
    h_vectors_test[3 * DIM + i] = h_vectors_base[73 * DIM + i];
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    ASSERT_EQ(table->size(stream), 0);

    copy_to_device(d_keys, h_keys_base, BASE_KEY_NUM);
    copy_to_device(d_scores, h_scores_base, BASE_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_base, BASE_KEY_NUM * DIM);
    S start_ts = host_nano<S>(stream);
    table->find_or_insert(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          nullptr, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    S end_ts = host_nano<S>(stream);
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);

    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    vector<S> sorted_scores(h_scores_temp);
    sort(sorted_scores.begin(), sorted_scores.end());
    ASSERT_GE(sorted_scores[0], start_ts);
    ASSERT_LE(sorted_scores[TEST_KEY_NUM - 1], end_ts);
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    table->find_or_insert(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          nullptr, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    start_ts = host_nano<S>(stream);
    table->assign(TEST_KEY_NUM, d_keys.as<K>(), nullptr, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    end_ts = host_nano<S>(stream);

    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    vector<S> updated_scores(TEST_KEY_NUM);
    int updated_count = 0;
    for (size_t j = 0; j < TEMP_KEY_NUM; ++j) {
      if (find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[j]) !=
          h_keys_test.end()) {
        ASSERT_GT(h_scores_temp[j], BUCKET_MAX_SIZE);
        updated_scores[updated_count++] = h_scores_temp[j];
      } else {
        ASSERT_LE(h_scores_temp[j], start_ts);
      }
    }
    sort(updated_scores.begin(), updated_scores.begin() + updated_count);
    ASSERT_GE(updated_scores[0], start_ts);
    ASSERT_LE(updated_scores[updated_count - 1], end_ts);
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_lfu_basic(size_t max_hbm_for_vectors,
                                   int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr int freq_range = 1000;

  auto options =
      make_options(max_hbm_for_vectors, key_start, INIT_CAPACITY, MAX_CAPACITY,
                   DIM, BUCKET_MAX_SIZE);
  using Table = TableT<EvictStrategy::kLfu>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);

  auto d_keys = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(V) * DIM);

  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
      freq_range);
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD, freq_range);

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  h_scores_test[2] = h_keys_base[72] % freq_range;
  h_scores_test[3] = h_keys_base[73] % freq_range;
  for (int i = 0; i < static_cast<int>(DIM); ++i) {
    h_vectors_test[2 * DIM + i] = h_vectors_base[72 * DIM + i];
    h_vectors_test[3 * DIM + i] = h_vectors_base[73 * DIM + i];
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    ASSERT_EQ(table->size(stream), 0);

    copy_to_device(d_keys, h_keys_base, BASE_KEY_NUM);
    copy_to_device(d_scores, h_scores_base, BASE_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_base, BASE_KEY_NUM * DIM);
    table->find_or_insert(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      ASSERT_EQ(h_scores_temp[j], h_keys_temp[j] % freq_range);
    }
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    table->assign(TEST_KEY_NUM, d_keys.as<K>(), d_scores.as<S>(), stream);
    table->find_or_insert(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      const bool in_base = find(h_keys_base.begin(), h_keys_base.end(),
                                h_keys_temp[j]) != h_keys_base.end();
      const bool in_test = find(h_keys_test.begin(), h_keys_test.end(),
                                h_keys_temp[j]) != h_keys_test.end();
      const S expected =
          (in_base && in_test) ? (h_keys_temp[j] % freq_range) * 3
                              : (h_keys_temp[j] % freq_range);
      ASSERT_EQ(h_scores_temp[j], expected);
    }
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_epochlru_basic(size_t max_hbm_for_vectors,
                                        int key_start = 0) {
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

  auto options =
      make_options(max_hbm_for_vectors, key_start, INIT_CAPACITY, MAX_CAPACITY,
                   DIM, BUCKET_MAX_SIZE);
  using Table = TableT<EvictStrategy::kEpochLru>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  auto d_keys = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(V) * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  for (int i = 0; i < static_cast<int>(DIM); ++i) {
    h_vectors_test[2 * DIM + i] = h_vectors_base[72 * DIM + i];
    h_vectors_test[3 * DIM + i] = h_vectors_base[73 * DIM + i];
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  S global_epoch = 1;
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    ASSERT_EQ(table->size(stream), 0);

    copy_to_device(d_keys, h_keys_base, BASE_KEY_NUM);
    copy_to_device(d_scores, h_scores_base, BASE_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_base, BASE_KEY_NUM * DIM);
    S start_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    table->set_global_epoch(global_epoch);
    table->find_or_insert(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          nullptr, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    S end_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    vector<S> sorted_scores(h_scores_temp);
    sort(sorted_scores.begin(), sorted_scores.end());
    ASSERT_GE(sorted_scores[0], ((global_epoch << 32) | start_ts));
    ASSERT_LE(sorted_scores[BASE_KEY_NUM - 1],
              ((global_epoch << 32) | end_ts));
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    ++global_epoch;
    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    start_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    table->set_global_epoch(global_epoch);
    table->assign(TEST_KEY_NUM, d_keys.as<K>(), nullptr, stream);
    table->find_or_insert(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          nullptr, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    end_ts = (host_nano<S>(stream) >> RSHIFT_ON_NANO) & 0xFFFFFFFF;
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    vector<S> updated_scores(TEST_KEY_NUM);
    int updated_count = 0;
    for (size_t j = 0; j < TEMP_KEY_NUM; ++j) {
      if (find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[j]) !=
          h_keys_test.end()) {
        ASSERT_GE(h_scores_temp[j], ((global_epoch << 32) | start_ts));
        updated_scores[updated_count++] = h_scores_temp[j];
      } else {
        ASSERT_LE(h_scores_temp[j], ((global_epoch << 32) | start_ts));
      }
    }
    sort(updated_scores.begin(), updated_scores.begin() + updated_count);
    ASSERT_GE(updated_scores[0], ((global_epoch << 32) | start_ts));
    ASSERT_LE(updated_scores[updated_count - 1],
              ((global_epoch << 32) | end_ts));
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_epochlfu_basic(size_t max_hbm_for_vectors,
                                        int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr int freq_range = 1000;

  auto options =
      make_options(max_hbm_for_vectors, key_start, INIT_CAPACITY, MAX_CAPACITY,
                   DIM, BUCKET_MAX_SIZE);
  using Table = TableT<EvictStrategy::kEpochLfu>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  auto d_keys = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(V) * DIM);

  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
      freq_range);
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD, freq_range);

  // Simulate overflow of low 32bits.
  h_scores_base[71] =
      static_cast<S>(numeric_limits<uint32_t>::max() - uint32_t{1});
  h_keys_test[1] = h_keys_base[71];
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  h_scores_test[1] = h_scores_base[71];
  h_scores_test[2] = h_keys_base[72] % freq_range;
  h_scores_test[3] = h_keys_base[73] % freq_range;
  for (int i = 0; i < static_cast<int>(DIM); ++i) {
    h_vectors_test[1 * DIM + i] = h_vectors_base[71 * DIM + i];
    h_vectors_test[2 * DIM + i] = h_vectors_base[72 * DIM + i];
    h_vectors_test[3 * DIM + i] = h_vectors_base[73 * DIM + i];
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  S global_epoch = 1;
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    ASSERT_EQ(table->size(stream), 0);

    copy_to_device(d_keys, h_keys_base, BASE_KEY_NUM);
    copy_to_device(d_scores, h_scores_base, BASE_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_base, BASE_KEY_NUM * DIM);
    table->set_global_epoch(global_epoch);
    table->find_or_insert(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      S original_score = (h_keys_temp[j] == h_keys_base[71])
                             ? h_scores_base[71]
                             : (h_keys_temp[j] % freq_range);
      ASSERT_EQ(h_scores_temp[j],
                make_expected_score_for_epochlfu<S>(global_epoch,
                                                    original_score));
    }
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    ++global_epoch;
    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    table->set_global_epoch(global_epoch);
    table->assign(TEST_KEY_NUM, d_keys.as<K>(), d_scores.as<S>(), stream);
    table->find_or_insert(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    ASSERT_TRUE(find(h_keys_temp.begin(), h_keys_temp.end(), h_keys_base[71]) !=
                h_keys_temp.end());
    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      const bool in_base = find(h_keys_base.begin(), h_keys_base.end(),
                                h_keys_temp[j]) != h_keys_base.end();
      const bool in_test = find(h_keys_test.begin(), h_keys_test.end(),
                                h_keys_temp[j]) != h_keys_test.end();
      S expected_score = 0;
      if (in_base && in_test) {
        expected_score =
            (h_keys_temp[j] == h_keys_base[71])
                ? make_expected_score_for_epochlfu<S>(global_epoch,
                                                      h_scores_base[71] * 2)
                : make_expected_score_for_epochlfu<S>(
                      global_epoch, (h_keys_temp[j] % freq_range) * 3);
      } else {
        expected_score =
            (h_keys_temp[j] == h_keys_base[71])
                ? make_expected_score_for_epochlfu<S>(
                      global_epoch - static_cast<S>(in_base), h_scores_base[71])
                : make_expected_score_for_epochlfu<S>(
                      global_epoch - static_cast<S>(in_base),
                      h_keys_temp[j] % freq_range);
      }
      ASSERT_EQ(h_scores_temp[j], expected_score);
    }
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_customized_basic(size_t max_hbm_for_vectors,
                                          int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 128;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 128;
  constexpr S base_score_start = 1000;
  constexpr S test_score_start = base_score_start + BASE_KEY_NUM;

  auto options =
      make_options(max_hbm_for_vectors, key_start, INIT_CAPACITY, MAX_CAPACITY,
                   DIM, BUCKET_MAX_SIZE);
  using Table = TableT<EvictStrategy::kCustomized>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  auto d_keys = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(V) * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);
  for (int i = 0; i < static_cast<int>(BASE_KEY_NUM); ++i) {
    h_scores_base[i] = base_score_start + i;
  }
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);
  for (int i = 0; i < static_cast<int>(TEST_KEY_NUM); ++i) {
    h_scores_test[i] = test_score_start + i;
  }
  for (int i = 64; i < static_cast<int>(TEST_KEY_NUM); ++i) {
    h_keys_test[i] = h_keys_base[i];
    for (int j = 0; j < static_cast<int>(DIM); ++j) {
      h_vectors_test[i * DIM + j] = h_vectors_base[i * DIM + j];
    }
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    ASSERT_EQ(table->size(stream), 0);
    copy_to_device(d_keys, h_keys_base, BASE_KEY_NUM);
    copy_to_device(d_scores, h_scores_base, BASE_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_base, BASE_KEY_NUM * DIM);
    table->find_or_insert(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    vector<S> sorted_scores(h_scores_temp);
    sort(sorted_scores.begin(), sorted_scores.end());
    auto expected_range = range_from<S, TEMP_KEY_NUM>(base_score_start);
    ASSERT_TRUE(equal(sorted_scores.begin(), sorted_scores.end(),
                      expected_range.begin()));
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    table->assign(TEST_KEY_NUM, d_keys.as<K>(), d_scores.as<S>(), stream);
    table->find_or_insert(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    sorted_scores = h_scores_temp;
    sort(sorted_scores.begin(), sorted_scores.end());
    auto expected_range_test = range_from<S, TEST_KEY_NUM>(test_score_start);
    ASSERT_TRUE(equal(sorted_scores.begin(), sorted_scores.end(),
                      expected_range_test.begin()));
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_customized_advanced(size_t max_hbm_for_vectors,
                                             int key_start = 0) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 8;
  constexpr uint64_t TEMP_KEY_NUM =
      (BASE_KEY_NUM > TEST_KEY_NUM) ? BASE_KEY_NUM : TEST_KEY_NUM;
  constexpr uint64_t TEST_TIMES = 256;
  constexpr S base_score_start = 1000;

  auto options =
      make_options(max_hbm_for_vectors, key_start, INIT_CAPACITY, MAX_CAPACITY,
                   DIM, BUCKET_MAX_SIZE);
  using Table = TableT<EvictStrategy::kCustomized>;

  vector<K> h_keys_base(BASE_KEY_NUM);
  vector<S> h_scores_base(BASE_KEY_NUM);
  vector<V> h_vectors_base(BASE_KEY_NUM * DIM);
  vector<K> h_keys_test(TEST_KEY_NUM);
  vector<S> h_scores_test(TEST_KEY_NUM);
  vector<V> h_vectors_test(TEST_KEY_NUM * DIM);
  vector<K> h_keys_temp(TEMP_KEY_NUM);
  vector<S> h_scores_temp(TEMP_KEY_NUM);
  vector<V> h_vectors_temp(TEMP_KEY_NUM * DIM);
  auto d_keys = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(K));
  auto d_scores = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(S));
  auto d_vectors = DeviceMem::alloc(TEMP_KEY_NUM * sizeof(V) * DIM);

  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);
  for (int i = 0; i < static_cast<int>(BASE_KEY_NUM); ++i) {
    h_scores_base[i] = base_score_start + i;
  }
  create_keys_in_one_buckets<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD);

  h_keys_test[4] = h_keys_base[72];
  h_keys_test[5] = h_keys_base[73];
  h_keys_test[6] = h_keys_base[74];
  h_keys_test[7] = h_keys_base[75];

  // replace four new keys to lower scores, would not be inserted.
  h_scores_test[0] = 20;
  h_scores_test[1] = 78;
  h_scores_test[2] = 97;
  h_scores_test[3] = 98;

  // replace four exist keys to new scores, just refresh the score for them.
  h_scores_test[4] = 99;
  h_scores_test[5] = 1010;
  h_scores_test[6] = 1020;
  h_scores_test[7] = 1035;

  for (int i = 4; i < static_cast<int>(TEST_KEY_NUM); ++i) {
    for (int j = 0; j < static_cast<int>(DIM); ++j) {
      h_vectors_test[i * DIM + j] =
          static_cast<V>(h_keys_test[i] * 0.00001);
    }
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    ASSERT_EQ(table->size(stream), 0);
    copy_to_device(d_keys, h_keys_base, BASE_KEY_NUM);
    copy_to_device(d_scores, h_scores_base, BASE_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_base, BASE_KEY_NUM * DIM);
    table->find_or_insert(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    vector<S> sorted_scores(h_scores_temp);
    sort(sorted_scores.begin(), sorted_scores.end());
    auto expected_range = range_from<S, TEMP_KEY_NUM>(base_score_start);
    ASSERT_TRUE(equal(sorted_scores.begin(), sorted_scores.end(),
                      expected_range.begin()));
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);

    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    table->assign(TEST_KEY_NUM, d_keys.as<K>(), d_scores.as<S>(), stream);
    table->find_or_insert(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          d_scores.as<S>(), stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);

    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      if (h_scores_temp[j] < base_score_start + 4) {
        ASSERT_EQ(find(h_keys_test.begin(), h_keys_test.begin() + 4,
                       h_keys_temp[j]),
                  h_keys_test.begin() + 4);
      }
      for (int k = 4; k < static_cast<int>(TEST_KEY_NUM); ++k) {
        if (h_keys_temp[j] == h_keys_test[k]) {
          ASSERT_EQ(h_scores_temp[j], h_scores_test[k]);
        }
      }
    }
    verify_values_match_keys(h_keys_temp, h_vectors_temp, BUCKET_MAX_SIZE, DIM);
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

template <typename K0, typename V0, typename S0, typename Table,
          size_t dim = 64>
void check_assign_on_epoch_lfu(Table* table, KVMSBuffer<K0, V0, S0>* data_buffer,
                           KVMSBuffer<K0, V0, S0>* evict_buffer,
                           KVMSBuffer<K0, V0, S0>* pre_data_buffer,
                           size_t len, aclrtStream stream,
                           unsigned int global_epoch) {
  map<K0, ValueArray<V0, dim>> values_before;
  map<K0, ValueArray<V0, dim>> values_after;
  unordered_map<K0, S0> scores_before;
  map<K0, S0> scores_after;
  map<K0, S0> current_batch_scores;
  map<K0, S0> current_evict_scores;

  for (size_t i = 0; i < len; ++i) {
    current_batch_scores[data_buffer->keys_ptr(false)[i]] =
        data_buffer->scores_ptr(false)[i];
  }

  const size_t table_size_before = table->size(stream);
  const size_t cap = table_size_before + len;
  HostAndDeviceBuffer<K0> tmp_keys;
  HostAndDeviceBuffer<V0> tmp_values;
  HostAndDeviceBuffer<S0> tmp_scores;
  tmp_keys.alloc(cap, stream);
  tmp_values.alloc(cap * dim, stream);
  tmp_scores.alloc(cap, stream);

  size_t exported = table->export_batch(table->capacity(), 0,
                                        tmp_keys.d_data, tmp_values.d_data,
                                        tmp_scores.d_data, stream);
  ASSERT_EQ(exported, table_size_before);
  tmp_keys.sync_data(false, stream);
  tmp_values.sync_data(false, stream);
  tmp_scores.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < table_size_before; ++i) {
    auto* vec =
        reinterpret_cast<ValueArray<V0, dim>*>(tmp_values.h_data + i * dim);
    values_before[tmp_keys.h_data[i]] = *vec;
    scores_before[tmp_keys.h_data[i]] = tmp_scores.h_data[i];
  }

  table->set_global_epoch(global_epoch);
  table->assign(len, data_buffer->keys_ptr(), data_buffer->scores_ptr(),
                stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  exported = table->export_batch(table->capacity(), 0, tmp_keys.d_data,
                                 tmp_values.d_data, tmp_scores.d_data, stream);
  ASSERT_EQ(exported, table_size_before);
  tmp_keys.sync_data(false, stream);
  tmp_values.sync_data(false, stream);
  tmp_scores.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  size_t score_error_cnt = 0;
  for (size_t i = 0; i < table_size_before; ++i) {
    auto* vec =
        reinterpret_cast<ValueArray<V0, dim>*>(tmp_values.h_data + i * dim);
    values_after[tmp_keys.h_data[i]] = *vec;
    scores_after[tmp_keys.h_data[i]] = tmp_scores.h_data[i];
  }

  for (const auto& it : current_batch_scores) {
    const K0 key = it.first;
    const S0 score = it.second;
    if (scores_before.find(key) != scores_before.end()) {
      const S0 current_score = scores_after[key];
      const S0 score_before = scores_before[key];
      const bool valid =
          ((current_score >> 32) == global_epoch) &&
          ((current_score & 0xFFFFFFFF) ==
           ((0xFFFFFFFF & score_before) + (0xFFFFFFFF & score)));
      if (!valid) {
        ++score_error_cnt;
      }
    }
  }
  ASSERT_EQ(score_error_cnt, 0);

  values_before = values_after;
  scores_before.clear();
  for (const auto& it : scores_after) {
    scores_before[it.first] = it.second;
  }
  values_after.clear();
  scores_after.clear();

  size_t filtered_len = table->insert_and_evict(
      len, data_buffer->keys_ptr(), data_buffer->values_ptr(),
      data_buffer->scores_ptr(), evict_buffer->keys_ptr(),
      evict_buffer->values_ptr(), evict_buffer->scores_ptr(), stream);
  evict_buffer->sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  for (size_t i = 0; i < filtered_len; ++i) {
    current_evict_scores[evict_buffer->keys_ptr(false)[i]] =
        evict_buffer->scores_ptr(false)[i];
  }

  const size_t table_size_after = table->size(stream);
  exported = table->export_batch(table->capacity(), 0, tmp_keys.d_data,
                                 tmp_values.d_data, tmp_scores.d_data, stream);
  ASSERT_EQ(exported, table_size_after);
  tmp_keys.sync_data(false, stream);
  tmp_values.sync_data(false, stream);
  tmp_scores.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt1 = 0;
  size_t score_error_cnt2 = 0;

  for (size_t i = 0; i < table_size_after; ++i) {
    auto* vec =
        reinterpret_cast<ValueArray<V0, dim>*>(tmp_values.h_data + i * dim);
    values_after[tmp_keys.h_data[i]] = *vec;
    scores_after[tmp_keys.h_data[i]] = tmp_scores.h_data[i];
  }

  for (size_t i = 0; i < filtered_len; ++i) {
    auto* vec = reinterpret_cast<ValueArray<V0, dim>*>(
        evict_buffer->values_ptr(false) + i * dim);
    values_after[evict_buffer->keys_ptr(false)[i]] = *vec;
    scores_after[evict_buffer->keys_ptr(false)[i]] =
        evict_buffer->scores_ptr(false)[i];
    if ((evict_buffer->scores_ptr(false)[i] >> 32) >= (global_epoch - 2)) {
      ++score_error_cnt1;
    }
  }

  for (const auto& it : current_batch_scores) {
    const K0 key = it.first;
    const S0 score = it.second;
    const S0 current_score = scores_after[key];
    S0 score_before = 0;
    if (values_after.find(key) != values_after.end() &&
        current_evict_scores.find(key) == current_evict_scores.end() &&
        scores_before.find(key) != scores_before.end()) {
      score_before = scores_before[key];
    }
    const bool valid =
        ((current_score >> 32) == global_epoch) &&
        ((current_score & 0xFFFFFFFF) ==
         ((0xFFFFFFFF & score_before) + (0xFFFFFFFF & score)));
    if (!valid) {
      ++score_error_cnt2;
    }
  }

  for (const auto& it : values_before) {
    if (values_after.find(it.first) == values_after.end()) {
      ++key_miss_cnt;
      continue;
    }
    const auto& vec0 = it.second;
    const auto& vec1 = values_after.at(it.first);
    for (size_t j = 0; j < dim; ++j) {
      if (vec0[j] != vec1[j]) {
        ++value_diff_cnt;
        break;
      }
    }
  }

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
  ASSERT_EQ(score_error_cnt1, 0);
  ASSERT_EQ(score_error_cnt2, 0);
}

void test_assign_advanced_on_epochlfu(size_t max_hbm_for_vectors) {
  const size_t U = 1024 * 1024;
  const size_t B = 100000;
  constexpr size_t dim = 16;

  TableOptions options;
  options.max_capacity = U;
  options.init_capacity = U;
  options.max_bucket_size = 128;
  options.max_hbm_for_vectors = GB(max_hbm_for_vectors);
  options.dim = dim;
  using Table = TableT<EvictStrategy::kEpochLfu>;

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  auto table = make_table<Table>(options);

  KVMSBuffer<K, V, S> evict_buffer;
  KVMSBuffer<K, V, S> data_buffer;
  KVMSBuffer<K, V, S> pre_data_buffer;
  evict_buffer.reserve(B, dim, stream);
  evict_buffer.to_zeros(stream);
  data_buffer.reserve(B, dim, stream);
  pre_data_buffer.reserve(B, dim, stream);

  int freq_range = 100;
  float repeat_rate = 0.9;
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
    data_buffer.sync_data(true, stream);
    if (global_epoch <= 1) {
      pre_data_buffer.copy_from(data_buffer, stream);
    }

    check_assign_on_epoch_lfu<K, V, S, Table, dim>(
        table.get(), &data_buffer, &evict_buffer, &pre_data_buffer, B, stream,
        global_epoch);

    pre_data_buffer.copy_from(data_buffer, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_evict_strategy_customized_correct_rate(size_t max_hbm_for_vectors,
                                                 int key_start = 0) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024UL;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr float expected_correct_rate = 0.964f;
  constexpr int rounds = 12;

  auto options =
      make_options(max_hbm_for_vectors, key_start, INIT_CAPACITY, MAX_CAPACITY,
                   DIM, MAX_BUCKET_SIZE);
  using Table = TableT<EvictStrategy::kCustomized>;

  vector<K> h_keys_base(BATCH_SIZE);
  vector<S> h_scores_base(BATCH_SIZE);
  vector<V> h_vectors_base(BATCH_SIZE * DIM);
  vector<K> h_keys_temp(MAX_CAPACITY);
  vector<S> h_scores_temp(MAX_CAPACITY);
  vector<V> h_vectors_temp(MAX_CAPACITY * DIM);
  auto d_keys = DeviceMem::alloc(MAX_CAPACITY * sizeof(K));
  auto d_scores = DeviceMem::alloc(MAX_CAPACITY * sizeof(S));
  auto d_vectors = DeviceMem::alloc(MAX_CAPACITY * sizeof(V) * DIM);

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  size_t global_start_key = 100000;
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    size_t start_key = global_start_key;
    ASSERT_EQ(table->size(stream), 0);

    for (int r = 0; r < rounds; ++r) {
      const size_t expected_min_key = global_start_key + INIT_CAPACITY * r;
      const size_t expected_max_key =
          global_start_key + INIT_CAPACITY * (r + 1) - 1;
      const size_t expected_table_size =
          (r == 0) ? size_t(expected_correct_rate * INIT_CAPACITY)
                   : INIT_CAPACITY;

      for (int s = 0; s < static_cast<int>(STEPS); ++s) {
        create_continuous_keys<K, S, V, DIM>(
            h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
            BATCH_SIZE, start_key);
        start_key += BATCH_SIZE;
        copy_to_device(d_keys, h_keys_base, BATCH_SIZE);
        copy_to_device(d_scores, h_scores_base, BATCH_SIZE);
        copy_to_device(d_vectors, h_vectors_base, BATCH_SIZE * DIM);
        table->assign(BATCH_SIZE, d_keys.as<K>(), d_scores.as<S>(), stream);
        table->find_or_insert(BATCH_SIZE, d_keys.as<K>(), d_vectors.as<V>(),
                              d_scores.as<S>(), stream);
        ACL_CHECK(aclrtSynchronizeStream(stream));
      }

      const size_t total_size = table->size(stream);
      ASSERT_GE(total_size, expected_table_size);
      ASSERT_EQ(MAX_CAPACITY, table->capacity());
      const size_t exported = table->export_batch(
          MAX_CAPACITY, 0, d_keys.as<K>(), d_vectors.as<V>(), d_scores.as<S>(),
          stream);
      copy_to_host(h_keys_temp, d_keys, MAX_CAPACITY);
      copy_to_host(h_scores_temp, d_scores, MAX_CAPACITY);
      copy_to_host(h_vectors_temp, d_vectors, MAX_CAPACITY * DIM);

      size_t bigger_score_counter = 0;
      K max_key = 0;
      size_t values_error_counter = 0;
      for (size_t j = 0; j < exported; ++j) {
        ASSERT_EQ(h_keys_temp[j], h_scores_temp[j]);
        max_key = max(max_key, h_keys_temp[j]);
        if (h_scores_temp[j] >= expected_min_key) {
          ++bigger_score_counter;
        }
        for (size_t k = 0; k < DIM; ++k) {
          if (h_vectors_temp[j * DIM + k] !=
              static_cast<float>(h_keys_temp[j] * 0.00001)) {
            ++values_error_counter;
          }
        }
      }
      ASSERT_EQ(values_error_counter, 0);
      const float correct_rate =
          (bigger_score_counter * 1.0f) / static_cast<float>(MAX_CAPACITY);
      cout << setprecision(3) << "[Round " << r
           << "]correct_rate=" << correct_rate << endl;
      ASSERT_GE(max_key, expected_max_key);
      ASSERT_GE(correct_rate, expected_correct_rate);
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

}  // namespace

TEST(AssignScoreTest, test_evict_strategy_lru_basic) {
  init_env();
  test_evict_strategy_lru_basic(16);
  test_evict_strategy_lru_basic(0, 34);
}

TEST(AssignScoreTest, test_evict_strategy_lfu_basic) {
  init_env();
  test_evict_strategy_lfu_basic(16);
  test_evict_strategy_lfu_basic(0, 2);
}

TEST(AssignScoreTest, test_evict_strategy_epochlru_basic) {
  init_env();
  test_evict_strategy_epochlru_basic(16, 51);
  test_evict_strategy_epochlru_basic(0);
}

TEST(AssignScoreTest, test_evict_strategy_epochlfu_basic) {
  init_env();
  test_evict_strategy_epochlfu_basic(16, 4);
  test_evict_strategy_epochlfu_basic(0);
}

TEST(AssignScoreTest, test_evict_strategy_customized_basic) {
  init_env();
  test_evict_strategy_customized_basic(16);
  test_evict_strategy_customized_basic(0, 11);
}

TEST(AssignScoreTest, test_evict_strategy_customized_advanced) {
  init_env();
  test_evict_strategy_customized_advanced(16, 33);
  test_evict_strategy_customized_advanced(0);
}

TEST(AssignScoreTest, test_assign_advanced_on_epochlfu) {
  init_env();
  test_assign_advanced_on_epochlfu(16);
}

TEST(AssignScoreTest, test_evict_strategy_customized_correct_rate) {
  init_env();
  // TODO: after blossom CI issue is resolved, the skip logic.
  const bool skip_hmem_check = (nullptr != std::getenv("IS_BLOSSOM_CI"));
  test_evict_strategy_customized_correct_rate(16, 44);
  if (!skip_hmem_check) {
    test_evict_strategy_customized_correct_rate(0);
  } else {
    cout << "The HMEM check is skipped in blossom CI!" << endl;
  }
}
