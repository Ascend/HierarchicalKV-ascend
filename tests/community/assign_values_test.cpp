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
#include <cstdint>
#include <cstdlib>
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

template <typename Table>
void export_table(Table* table, size_t expected_count, DeviceMem& keys,
                  DeviceMem& values, DeviceMem& scores, vector<K>& host_keys,
                  vector<V>& host_values, vector<S>& host_scores,
                  aclrtStream stream) {
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
    h_vectors_test[2 * DIM + i] =
        static_cast<float>(h_keys_base[72] * 0.00002);
    h_vectors_test[3 * DIM + i] =
        static_cast<float>(h_keys_base[73] * 0.00002);
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  for (int i = 0; i < static_cast<int>(TEST_TIMES); ++i) {
    auto table = make_table<Table>(options);
    ASSERT_EQ(table->size(stream), 0);

    copy_to_device(d_keys, h_keys_base, BASE_KEY_NUM);
    copy_to_device(d_scores, h_scores_base, BASE_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_base, BASE_KEY_NUM * DIM);
    S start_ts = npu::hkv::host_nano<S>(stream);
    table->find_or_insert(BASE_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                          nullptr, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    S end_ts = npu::hkv::host_nano<S>(stream);
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);

    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    vector<S> sorted_scores(h_scores_temp);
    sort(sorted_scores.begin(), sorted_scores.end());
    ASSERT_GE(sorted_scores[0], start_ts);
    ASSERT_LE(sorted_scores[TEST_KEY_NUM - 1], end_ts);
    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      for (size_t k = 0; k < DIM; ++k) {
        ASSERT_EQ(h_vectors_temp[j * DIM + k],
                  static_cast<float>(h_keys_temp[j] * 0.00001));
      }
    }

    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    start_ts = npu::hkv::host_nano<S>(stream);
    table->assign_values(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                         stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);
    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      const V expected_v = (h_keys_temp[j] == h_keys_test[2] ||
                            h_keys_temp[j] == h_keys_test[3])
                               ? static_cast<V>(h_keys_temp[j] * 0.00002)
                               : static_cast<V>(h_keys_temp[j] * 0.00001);
      for (size_t k = 0; k < DIM; ++k) {
        ASSERT_EQ(h_vectors_temp[j * DIM + k], expected_v);
      }
      ASSERT_LE(h_scores_temp[j], start_ts);
    }
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

  constexpr int freq_range = 1000;
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_base.data(), h_scores_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF,
      freq_range);
  create_keys_in_one_buckets_lfu<K, S, V, DIM>(
      h_keys_test.data(), h_scores_test.data(), h_vectors_test.data(),
      TEST_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0x3FFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFD, freq_range);

  // Simulate overflow of low 32bits.
  h_scores_base[71] = static_cast<S>(numeric_limits<uint32_t>::max() -
                                     static_cast<uint32_t>(1));

  h_keys_test[1] = h_keys_base[71];
  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];
  h_scores_test[1] = h_scores_base[71];
  h_scores_test[2] = h_keys_base[72] % freq_range;
  h_scores_test[3] = h_keys_base[73] % freq_range;
  for (int i = 0; i < static_cast<int>(DIM); ++i) {
    h_vectors_test[1 * DIM + i] =
        static_cast<float>(h_keys_base[71] * 0.00002);
    h_vectors_test[2 * DIM + i] =
        static_cast<float>(h_keys_base[72] * 0.00002);
    h_vectors_test[3 * DIM + i] =
        static_cast<float>(h_keys_base[73] * 0.00002);
  }

  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream));
  constexpr S global_epoch = 1;
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
      const S original_score =
          (h_keys_temp[j] == h_keys_base[71])
              ? h_scores_base[71]
              : static_cast<S>(h_keys_temp[j] % freq_range);
      const S expected_score =
          make_expected_score_for_epochlfu<S>(global_epoch, original_score);
      ASSERT_EQ(h_scores_temp[j], expected_score);
      for (size_t k = 0; k < DIM; ++k) {
        ASSERT_EQ(h_vectors_temp[j * DIM + k],
                  static_cast<float>(h_keys_temp[j] * 0.00001));
      }
    }

    copy_to_device(d_keys, h_keys_test, TEST_KEY_NUM);
    copy_to_device(d_scores, h_scores_test, TEST_KEY_NUM);
    copy_to_device(d_vectors, h_vectors_test, TEST_KEY_NUM * DIM);
    table->assign_values(TEST_KEY_NUM, d_keys.as<K>(), d_vectors.as<V>(),
                         stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table->size(stream), BUCKET_MAX_SIZE);

    export_table(table.get(), BUCKET_MAX_SIZE, d_keys, d_vectors, d_scores,
                 h_keys_temp, h_vectors_temp, h_scores_temp, stream);
    ASSERT_TRUE(h_keys_temp.end() !=
                find(h_keys_temp.begin(), h_keys_temp.end(), h_keys_base[71]));

    for (size_t j = 0; j < BUCKET_MAX_SIZE; ++j) {
      const S original_score =
          (h_keys_temp[j] == h_keys_base[71])
              ? h_scores_base[71]
              : static_cast<S>(h_keys_temp[j] % freq_range);
      const S expected_score =
          make_expected_score_for_epochlfu<S>(global_epoch, original_score);
      ASSERT_EQ(h_scores_temp[j], expected_score);

      const V expected_v = (h_keys_temp[j] == h_keys_test[1] ||
                            h_keys_temp[j] == h_keys_test[2] ||
                            h_keys_temp[j] == h_keys_test[3])
                               ? static_cast<V>(h_keys_temp[j] * 0.00002)
                               : static_cast<V>(h_keys_temp[j] * 0.00001);
      for (size_t k = 0; k < DIM; ++k) {
        ASSERT_EQ(h_vectors_temp[j * DIM + k], expected_v);
      }
    }
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

template <typename K0, typename V0, typename S0, typename Table,
          size_t dim = 64>
void check_assign_values_on_epoch_lfu(
    Table* table, KVMSBuffer<K0, V0, S0>* data_buffer,
    KVMSBuffer<K0, V0, S0>* evict_buffer,
    KVMSBuffer<K0, V0, S0>* pre_data_buffer, size_t len, aclrtStream stream,
    unsigned int global_epoch) {
  (void)pre_data_buffer;
  map<K0, ValueArray<V0, dim>> values_map_before_insert;
  map<K0, ValueArray<V0, dim>> values_map_after_insert;
  unordered_map<K0, S0> scores_map_before_insert;
  map<K0, S0> scores_map_after_insert;
  map<K0, S0> scores_map_current_batch;
  map<K0, S0> scores_map_current_evict;

  for (size_t i = 0; i < len; ++i) {
    scores_map_current_batch[data_buffer->keys_ptr(false)[i]] =
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

  size_t exported = table->export_batch(table->capacity(), 0, tmp_keys.d_data,
                                        tmp_values.d_data, tmp_scores.d_data,
                                        stream);
  ASSERT_EQ(exported, table_size_before);

  tmp_keys.sync_data(false, stream);
  tmp_values.sync_data(false, stream);
  tmp_scores.sync_data(false, stream);
  ACL_CHECK(aclrtMemcpyAsync(tmp_keys.h_data + table_size_before,
                             len * sizeof(K0), data_buffer->keys_ptr(),
                             len * sizeof(K0),
                             ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK(aclrtMemcpyAsync(tmp_values.h_data + table_size_before * dim,
                             len * dim * sizeof(V0),
                             data_buffer->values_ptr(),
                             len * dim * sizeof(V0),
                             ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK(aclrtMemcpyAsync(tmp_scores.h_data + table_size_before,
                             len * sizeof(S0), data_buffer->scores_ptr(),
                             len * sizeof(S0),
                             ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < cap; ++i) {
    auto* vec =
        reinterpret_cast<ValueArray<V0, dim>*>(tmp_values.h_data + i * dim);
    values_map_before_insert[tmp_keys.h_data[i]] = *vec;
  }

  for (size_t i = 0; i < table_size_before; ++i) {
    scores_map_before_insert[tmp_keys.h_data[i]] = tmp_scores.h_data[i];
  }

  table->assign_values(len, data_buffer->keys_ptr(),
                       data_buffer->values_ptr(), stream);
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
    values_map_after_insert[tmp_keys.h_data[i]] = *vec;
    scores_map_after_insert[tmp_keys.h_data[i]] = tmp_scores.h_data[i];
  }

  for (const auto& it : scores_map_current_batch) {
    const K0 key = it.first;
    if (scores_map_before_insert.find(key) != scores_map_before_insert.end()) {
      const S0 current_score = scores_map_after_insert[key];
      const S0 score_before_insert = scores_map_before_insert[key];
      const bool valid =
          ((current_score >> 32) < global_epoch) &&
          ((current_score & 0xFFFFFFFF) ==
           (0xFFFFFFFF & score_before_insert));
      if (!valid) {
        ++score_error_cnt;
      }
    }
  }
  cout << "Check assign_values behavior got "
       << ", score_error_cnt: " << score_error_cnt << ", while len: " << len
       << endl;
  ASSERT_EQ(score_error_cnt, 0);

  for (size_t i = 0; i < table_size_before; ++i) {
    values_map_before_insert[tmp_keys.h_data[i]] =
        values_map_after_insert[tmp_keys.h_data[i]];
    scores_map_before_insert[tmp_keys.h_data[i]] =
        scores_map_after_insert[tmp_keys.h_data[i]];
  }
  values_map_after_insert.clear();
  scores_map_after_insert.clear();

  table->set_global_epoch(global_epoch);
  const size_t filtered_len = table->insert_and_evict(
      len, data_buffer->keys_ptr(), data_buffer->values_ptr(),
      data_buffer->scores_ptr(), evict_buffer->keys_ptr(),
      evict_buffer->values_ptr(), evict_buffer->scores_ptr(), stream);
  evict_buffer->sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < filtered_len; ++i) {
    scores_map_current_evict[evict_buffer->keys_ptr(false)[i]] =
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

  for (size_t i = 0; i < filtered_len; ++i) {
    auto* vec = reinterpret_cast<ValueArray<V0, dim>*>(
        evict_buffer->values_ptr(false) + i * dim);
    values_map_after_insert[evict_buffer->keys_ptr(false)[i]] = *vec;
    scores_map_after_insert[evict_buffer->keys_ptr(false)[i]] =
        evict_buffer->scores_ptr(false)[i];
    if ((evict_buffer->scores_ptr(false)[i] >> 32) >= (global_epoch - 2)) {
      ++score_error_cnt1;
    }
  }

  for (size_t i = 0; i < table_size_after; ++i) {
    auto* vec =
        reinterpret_cast<ValueArray<V0, dim>*>(tmp_values.h_data + i * dim);
    values_map_after_insert[tmp_keys.h_data[i]] = *vec;
    scores_map_after_insert[tmp_keys.h_data[i]] = tmp_scores.h_data[i];
  }

  for (const auto& it : scores_map_current_batch) {
    const K0 key = it.first;
    const S0 score = it.second;
    const S0 current_score = scores_map_after_insert[key];
    S0 score_before_insert = 0;
    if (values_map_after_insert.find(key) != values_map_after_insert.end() &&
        scores_map_current_evict.find(key) == scores_map_current_evict.end() &&
        scores_map_before_insert.find(key) != scores_map_before_insert.end()) {
      score_before_insert = scores_map_before_insert[key];
    }
    const bool valid =
        ((current_score >> 32) == global_epoch) &&
        ((current_score & 0xFFFFFFFF) ==
         ((0xFFFFFFFF & score_before_insert) + (0xFFFFFFFF & score)));
    if (!valid) {
      ++score_error_cnt2;
    }
  }

  for (const auto& it : values_map_before_insert) {
    if (values_map_after_insert.find(it.first) ==
        values_map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    const auto& vec0 = it.second;
    const auto& vec1 = values_map_after_insert.at(it.first);
    for (size_t j = 0; j < dim; ++j) {
      if (vec0[j] != vec1[j]) {
        ++value_diff_cnt;
        break;
      }
    }
  }

  cout << "Check insert_and_evict behavior got "
       << "key_miss_cnt: " << key_miss_cnt
       << ", value_diff_cnt: " << value_diff_cnt
       << ", score_error_cnt1: " << score_error_cnt1
       << ", score_error_cnt2: " << score_error_cnt2
       << ", while table_size_before: " << table_size_before
       << ", while table_size_after: " << table_size_after
       << ", while len: " << len << endl;

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

  constexpr int freq_range = 100;
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
    data_buffer.sync_data(true, stream);
    if (global_epoch <= 1) {
      pre_data_buffer.copy_from(data_buffer, stream);
    }

    check_assign_values_on_epoch_lfu<K, V, S, Table, dim>(
        table.get(), &data_buffer, &evict_buffer, &pre_data_buffer, B, stream,
        global_epoch);

    pre_data_buffer.copy_from(data_buffer, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }
  ACL_CHECK(aclrtDestroyStream(stream));
}

}  // namespace

TEST(AssignValuesTest, test_evict_strategy_lru_basic) {
  init_env();
  test_evict_strategy_lru_basic(16, 21);
  test_evict_strategy_lru_basic(0);
}

TEST(AssignValuesTest, test_evict_strategy_epochlfu_basic) {
  init_env();
  test_evict_strategy_epochlfu_basic(16);
  test_evict_strategy_epochlfu_basic(0, 8);
}

TEST(AssignValuesTest, test_assign_advanced_on_epochlfu) {
  init_env();
  test_assign_advanced_on_epochlfu(16);
}
