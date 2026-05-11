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
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace community_test_util;
using namespace npu::hkv;

namespace {

constexpr size_t dim = 64;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

class InsertAndEvictTest : public ::testing::Test {
 protected:
  void SetUp() override { init_env(); }
};

template <typename K, typename V>
void emplace_value_map(std::map<K, ValueArray<V, dim>>& dst, const K* keys,
                       const V* values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const auto* vec =
        reinterpret_cast<const ValueArray<V, dim>*>(values + i * dim);
    dst[keys[i]] = *vec;
  }
}

template <typename K, typename V, typename S, typename Table>
void capture_table_snapshot(Table* table, size_t cap,
                            HostAndDeviceBuffer<K>& d_keys,
                            HostAndDeviceBuffer<V>& d_values,
                            HostAndDeviceBuffer<S>& d_scores,
                            size_t* exported_count, aclrtStream stream) {
  d_keys.alloc(cap, stream);
  d_values.alloc(cap * dim, stream);
  d_scores.alloc(cap, stream);
  d_keys.to_zeros(stream);
  d_values.to_zeros(stream);
  d_scores.to_zeros(stream);
  *exported_count = table->export_batch(table->capacity(), 0, d_keys.d_data,
                                        d_values.d_data, d_scores.d_data,
                                        stream);
  d_keys.sync_data(false, stream);
  d_values.sync_data(false, stream);
  d_scores.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
}

/*
 * There are several steps to check whether if
 * the insert_and_evict API is safe to use:
 *
 *   step1: create a table with max_capacity U
 *   step2: Insert M keys into table while M < U. And
 *     the table size became m <= M. M - m keys was
 *     evicted.
 *   step3: Insert N keys into table while m + N > U, with
 *     no same key with M keys. And p keys gets evicted.
 *     If now the table size is v. Then total number of
 *     keys T = v + p + M - m, must equal to VT = M + N,
 *     while the keys, values, and scores match.
 *   step4: export table and check all values.
 */
void test_insert_and_evict_basic() {
  HashTableOptions opt{};

  // table setting
  const size_t init_capacity = 1024;

  // numeric setting
  const size_t U = 2llu << 18;
  const size_t M = (U >> 1);
  const size_t N = (U >> 1) + 17;  // Add a prime to test the non-aligned case.

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.num_of_buckets_per_alloc = 8;
  opt.dim = dim;
  opt.io_by_cpu = false;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kCustomized>;

  std::map<i64, ValueArray<f32, dim>> summarized_kvs;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  // step1
  auto table = std::make_unique<Table>();
  table->init(opt);

  // step2
  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(M, dim, stream);
  evict_buffer.to_zeros(stream);

  KVMSBuffer<i64, f32, u64> buffer;
  buffer.reserve(M, dim, stream);
  buffer.to_range(0, 1, stream);
  buffer.set_score(static_cast<u64>(1), stream);
  size_t n_evicted = table->insert_and_evict(
      M, buffer.keys_ptr(), buffer.values_ptr(), buffer.scores_ptr(),
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(),
      evict_buffer.scores_ptr(), stream);
  const size_t table_size_m = table->size(stream);
  buffer.sync_data(false, stream);
  evict_buffer.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ASSERT_EQ(n_evicted + table_size_m, M);
  emplace_value_map(summarized_kvs, evict_buffer.keys_ptr(false),
                    evict_buffer.values_ptr(false), n_evicted);

  // step3
  evict_buffer.reserve(N, dim, stream);
  buffer.reserve(N, dim, stream);
  buffer.to_range(M, 1, stream);
  buffer.set_score(static_cast<u64>(2), stream);
  n_evicted = table->insert_and_evict(
      N, buffer.keys_ptr(), buffer.values_ptr(), buffer.scores_ptr(),
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(),
      evict_buffer.scores_ptr(), stream);
  const size_t table_size_n = table->size(stream);
  buffer.sync_data(false, stream);
  evict_buffer.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ASSERT_EQ(table_size_m + N, table_size_n + n_evicted);
  emplace_value_map(summarized_kvs, evict_buffer.keys_ptr(false),
                    evict_buffer.values_ptr(false), n_evicted);

  // step4
  buffer.reserve(table_size_n, dim, stream);
  const size_t n_exported = table->export_batch(table->capacity(), 0,
                                                buffer.keys_ptr(),
                                                buffer.values_ptr(),
                                                buffer.scores_ptr(), stream);
  ASSERT_EQ(table_size_n, n_exported);
  buffer.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  emplace_value_map(summarized_kvs, buffer.keys_ptr(false),
                    buffer.values_ptr(false), n_exported);

  size_t k = 0;
  for (const auto& item : summarized_kvs) {
    ASSERT_EQ(item.first, static_cast<i64>(k));
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(item.second.data[j], static_cast<f32>(k));
    }
    ++k;
  }
  ASSERT_EQ(summarized_kvs.size(), M + N);

  ACL_CHECK(aclrtDestroyStream(stream));
}

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvict(Table* table, KVMSBuffer<K, V, S>* data_buffer,
                         KVMSBuffer<K, V, S>* evict_buffer, size_t len,
                         aclrtStream stream) {
  std::map<K, ValueArray<V, dim>> map_before_insert;
  std::map<K, ValueArray<V, dim>> map_after_insert;

  const size_t table_size_before = table->size(stream);
  const size_t cap = table_size_before + len;

  HostAndDeviceBuffer<K> d_tmp_keys;
  HostAndDeviceBuffer<V> d_tmp_values;
  HostAndDeviceBuffer<S> d_tmp_scores;
  HostAndDeviceBuffer<bool> d_tmp_founds;

  size_t table_size_verify0 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify0, stream);
  ASSERT_EQ(table_size_before, table_size_verify0);

  emplace_value_map(map_before_insert, d_tmp_keys.h_data, d_tmp_values.h_data,
                    table_size_before);
  emplace_value_map(map_before_insert, data_buffer->keys_ptr(false),
                    data_buffer->values_ptr(false), len);

  auto start = std::chrono::steady_clock::now();
  const size_t filtered_len = table->insert_and_evict(
      len, data_buffer->keys_ptr(), data_buffer->values_ptr(),
      (Table::evict_strategy == EvictStrategy::kLru ||
       Table::evict_strategy == EvictStrategy::kEpochLru)
          ? nullptr
          : data_buffer->scores_ptr(),
      evict_buffer->keys_ptr(), evict_buffer->values_ptr(),
      evict_buffer->scores_ptr(), stream);
  evict_buffer->sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  auto end = std::chrono::steady_clock::now();
  const auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  {
    HostAndDeviceBuffer<V> d_find_values;
    HostAndDeviceBuffer<S> d_find_scores;
    d_find_values.alloc(len * dim, stream);
    d_find_scores.alloc(len, stream);
    d_tmp_founds.alloc(len, stream);
    d_tmp_founds.to_zeros(stream);
    table->find(len, data_buffer->keys_ptr(), d_find_values.d_data,
                d_tmp_founds.d_data, d_find_scores.d_data, stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t found_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++found_counter;
      }
    }
    std::cout << "filtered_len:" << filtered_len
              << ", miss counter:" << len - found_counter << std::endl;

    d_tmp_founds.to_zeros(stream);
    table->contains(len, data_buffer->keys_ptr(), d_tmp_founds.d_data, stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t contains_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++contains_counter;
      }
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  const float dur = static_cast<float>(diff.count());

  const size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify1, stream);
  ASSERT_EQ(table_size_verify1, table_size_after);

  const size_t new_cap = table_size_after + filtered_len;
  std::vector<K> combined_keys(new_cap);
  std::vector<V> combined_values(new_cap * dim);
  std::copy_n(d_tmp_keys.h_data, table_size_after, combined_keys.begin());
  std::copy_n(d_tmp_values.h_data, table_size_after * dim, combined_values.begin());
  std::copy_n(evict_buffer->keys_ptr(false), filtered_len,
              combined_keys.begin() + table_size_after);
  std::copy_n(evict_buffer->values_ptr(false), filtered_len * dim,
              combined_values.begin() + table_size_after * dim);

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  for (int64_t i = static_cast<int64_t>(new_cap) - 1; i >= 0; --i) {
    const auto* vec = reinterpret_cast<const ValueArray<V, dim>*>(
        combined_values.data() + i * dim);
    map_after_insert[combined_keys[static_cast<size_t>(i)]] = *vec;
  }

  for (const auto& item : map_before_insert) {
    const auto after_it = map_after_insert.find(item.first);
    if (after_it == map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    for (size_t j = 0; j < dim; ++j) {
      if (item.second.data[j] != after_it->second.data[j]) {
        ++value_diff_cnt;
        break;
      }
    }
  }
  std::cout << "Check insert_and_evict behavior got "
            << "key_miss_cnt: " << key_miss_cnt
            << ", value_diff_cnt: " << value_diff_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
}

void test_insert_and_evict_advanced_on_lru() {
  const size_t U = 524288;
  const size_t init_capacity = U;
  const size_t B = 524288 + 13;

  HashTableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  opt.num_of_buckets_per_alloc = 32;
  opt.dim = dim;
  opt.io_by_cpu = false;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kLru>;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  auto table = std::make_unique<Table>();
  table->init(opt);

  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(B, dim, stream);
  evict_buffer.to_zeros(stream);

  KVMSBuffer<i64, f32, u64> data_buffer;
  data_buffer.reserve(B, dim, stream);

  for (int i = 0; i < 16; ++i) {
    create_random_keys<i64, u64, f32, dim>(
        data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), B, B * 16);
    data_buffer.sync_data(true, stream);

    CheckInsertAndEvict<i64, f32, u64, Table>(table.get(), &data_buffer,
                                              &evict_buffer, B, stream);
  }

  ACL_CHECK(aclrtDestroyStream(stream));
}

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvictOnLfu(Table* table,
                              KVMSBuffer<K, V, S>* data_buffer,
                              KVMSBuffer<K, V, S>* evict_buffer, size_t len,
                              aclrtStream stream, unsigned int global_epoch) {
  std::map<K, ValueArray<V, dim>> values_map_before_insert;
  std::map<K, ValueArray<V, dim>> values_map_after_insert;
  std::unordered_map<K, S> scores_map_before_insert;
  std::map<K, S> scores_map_after_insert;
  std::map<K, S> scores_map_current_batch;
  std::map<K, S> scores_map_current_evict;

  for (size_t i = 0; i < len; ++i) {
    scores_map_current_batch[data_buffer->keys_ptr(false)[i]] =
        data_buffer->scores_ptr(false)[i];
  }

  const size_t table_size_before = table->size(stream);
  const size_t cap = table_size_before + len;

  HostAndDeviceBuffer<K> d_tmp_keys;
  HostAndDeviceBuffer<V> d_tmp_values;
  HostAndDeviceBuffer<S> d_tmp_scores;
  HostAndDeviceBuffer<bool> d_tmp_founds;

  size_t table_size_verify0 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify0, stream);
  ASSERT_EQ(table_size_before, table_size_verify0);

  emplace_value_map(values_map_before_insert, d_tmp_keys.h_data,
                    d_tmp_values.h_data, table_size_before);
  emplace_value_map(values_map_before_insert, data_buffer->keys_ptr(false),
                    data_buffer->values_ptr(false), len);
  for (size_t i = 0; i < table_size_before; ++i) {
    scores_map_before_insert[d_tmp_keys.h_data[i]] = d_tmp_scores.h_data[i];
  }

  auto start = std::chrono::steady_clock::now();
  table->set_global_epoch(global_epoch);
  const size_t filtered_len = table->insert_and_evict(
      len, data_buffer->keys_ptr(), data_buffer->values_ptr(),
      data_buffer->scores_ptr(), evict_buffer->keys_ptr(),
      evict_buffer->values_ptr(), evict_buffer->scores_ptr(), stream);
  evict_buffer->sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  auto end = std::chrono::steady_clock::now();
  const auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  {
    HostAndDeviceBuffer<V> d_find_values;
    HostAndDeviceBuffer<S> d_find_scores;
    d_find_values.alloc(len * dim, stream);
    d_find_scores.alloc(len, stream);
    d_tmp_founds.alloc(len, stream);
    d_tmp_founds.to_zeros(stream);
    table->find(len, data_buffer->keys_ptr(), d_find_values.d_data,
                d_tmp_founds.d_data, d_find_scores.d_data, stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t found_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++found_counter;
      }
    }

    d_tmp_founds.to_zeros(stream);
    table->contains(len, data_buffer->keys_ptr(), d_tmp_founds.d_data, stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t contains_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++contains_counter;
      }
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  for (size_t i = 0; i < filtered_len; ++i) {
    scores_map_current_evict[evict_buffer->keys_ptr(false)[i]] =
        evict_buffer->scores_ptr(false)[i];
  }

  const float dur = static_cast<float>(diff.count());
  const size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify1, stream);
  ASSERT_EQ(table_size_verify1, table_size_after);

  const size_t new_cap = table_size_after + filtered_len;
  std::vector<K> combined_keys(new_cap);
  std::vector<V> combined_values(new_cap * dim);
  std::vector<S> combined_scores(new_cap);
  std::copy_n(d_tmp_keys.h_data, table_size_after, combined_keys.begin());
  std::copy_n(d_tmp_values.h_data, table_size_after * dim, combined_values.begin());
  std::copy_n(d_tmp_scores.h_data, table_size_after, combined_scores.begin());
  std::copy_n(evict_buffer->keys_ptr(false), filtered_len,
              combined_keys.begin() + table_size_after);
  std::copy_n(evict_buffer->values_ptr(false), filtered_len * dim,
              combined_values.begin() + table_size_after * dim);
  std::copy_n(evict_buffer->scores_ptr(false), filtered_len,
              combined_scores.begin() + table_size_after);

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt = 0;

  for (int64_t i = static_cast<int64_t>(new_cap) - 1; i >= 0; --i) {
    const auto* vec = reinterpret_cast<const ValueArray<V, dim>*>(
        combined_values.data() + i * dim);
    values_map_after_insert[combined_keys[static_cast<size_t>(i)]] = *vec;
    scores_map_after_insert[combined_keys[static_cast<size_t>(i)]] =
        combined_scores[static_cast<size_t>(i)];
  }

  for (const auto& item : scores_map_current_batch) {
    const K key = item.first;
    const S current_score = scores_map_after_insert[key];
    S score_before_insert = 0;
    if (scores_map_before_insert.find(key) != scores_map_before_insert.end() &&
        scores_map_current_evict.find(key) == scores_map_current_evict.end()) {
      score_before_insert = scores_map_before_insert[key];
    } else {
      continue;
    }
    if (current_score != item.second + score_before_insert) {
      ++score_error_cnt;
    }
  }

  ASSERT_EQ(values_map_before_insert.size(), values_map_after_insert.size());
  for (const auto& item : values_map_before_insert) {
    const auto after_it = values_map_after_insert.find(item.first);
    if (after_it == values_map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    for (size_t j = 0; j < dim; ++j) {
      if (item.second.data[j] != after_it->second.data[j]) {
        ++value_diff_cnt;
      }
    }
  }
  std::cout << "Check insert_and_evict behavior got "
            << "key_miss_cnt: " << key_miss_cnt
            << ", value_diff_cnt: " << value_diff_cnt
            << ", score_error_cnt: " << score_error_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(score_error_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
}

void test_insert_and_evict_advanced_on_lfu() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 256 * 1024;

  HashTableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  opt.num_of_buckets_per_alloc = 32;
  opt.dim = dim;
  opt.io_by_cpu = false;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kLfu>;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  auto table = std::make_unique<Table>();
  table->init(opt);

  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(B, dim, stream);
  evict_buffer.to_zeros(stream);

  KVMSBuffer<i64, f32, u64> data_buffer;
  data_buffer.reserve(B, dim, stream);

  for (unsigned int global_epoch = 1; global_epoch <= 32; ++global_epoch) {
    create_random_keys_advanced<i64, u64, f32>(
        dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), B, B * 16, 100);
    data_buffer.sync_data(true, stream);

    CheckInsertAndEvictOnLfu<i64, f32, u64, Table>(table.get(), &data_buffer,
                                                   &evict_buffer, B, stream,
                                                   global_epoch);
  }

  ACL_CHECK(aclrtDestroyStream(stream));
}

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvictOnEpochLru(Table* table,
                                   KVMSBuffer<K, V, S>* data_buffer,
                                   KVMSBuffer<K, V, S>* evict_buffer,
                                   size_t len, aclrtStream stream,
                                   unsigned int global_epoch) {
  std::map<K, ValueArray<V, dim>> values_map_before_insert;
  std::map<K, ValueArray<V, dim>> values_map_after_insert;
  std::map<K, S> scores_map_before_insert;
  std::map<K, S> scores_map_after_insert;
  std::map<K, S> scores_map_current_batch;

  for (size_t i = 0; i < len; ++i) {
    scores_map_current_batch[data_buffer->keys_ptr(false)[i]] =
        data_buffer->scores_ptr(false)[i];
  }

  const size_t table_size_before = table->size(stream);
  const size_t cap = table_size_before + len;

  HostAndDeviceBuffer<K> d_tmp_keys;
  HostAndDeviceBuffer<V> d_tmp_values;
  HostAndDeviceBuffer<S> d_tmp_scores;
  HostAndDeviceBuffer<bool> d_tmp_founds;

  size_t table_size_verify0 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify0, stream);
  ASSERT_EQ(table_size_before, table_size_verify0);

  emplace_value_map(values_map_before_insert, d_tmp_keys.h_data,
                    d_tmp_values.h_data, table_size_before);
  emplace_value_map(values_map_before_insert, data_buffer->keys_ptr(false),
                    data_buffer->values_ptr(false), len);
  for (size_t i = 0; i < table_size_before; ++i) {
    scores_map_before_insert[d_tmp_keys.h_data[i]] = d_tmp_scores.h_data[i];
  }

  const S nano_before_insert = host_nano<S>(stream);

  auto start = std::chrono::steady_clock::now();
  table->set_global_epoch(global_epoch);
  const size_t filtered_len = table->insert_and_evict(
      len, data_buffer->keys_ptr(), data_buffer->values_ptr(), nullptr,
      evict_buffer->keys_ptr(), evict_buffer->values_ptr(),
      evict_buffer->scores_ptr(), stream);
  evict_buffer->sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  auto end = std::chrono::steady_clock::now();
  const auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  const S nano_after_insert = host_nano<S>(stream);

  {
    HostAndDeviceBuffer<V> d_find_values;
    HostAndDeviceBuffer<S> d_find_scores;
    d_find_values.alloc(len * dim, stream);
    d_find_scores.alloc(len, stream);
    d_tmp_founds.alloc(len, stream);
    d_tmp_founds.to_zeros(stream);
    table->find(len, data_buffer->keys_ptr(), d_find_values.d_data,
                d_tmp_founds.d_data, d_find_scores.d_data, stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t found_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++found_counter;
      }
    }
    std::cout << "filtered_len:" << filtered_len
              << ", miss counter:" << len - found_counter << std::endl;
    ASSERT_EQ(len, found_counter);

    d_tmp_founds.to_zeros(stream);
    table->contains(len, data_buffer->keys_ptr(), d_tmp_founds.d_data, stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t contains_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++contains_counter;
      }
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  const float dur = static_cast<float>(diff.count());
  const size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify1, stream);
  ASSERT_EQ(table_size_verify1, table_size_after);

  const size_t new_cap = table_size_after + filtered_len;
  std::vector<K> combined_keys(new_cap);
  std::vector<V> combined_values(new_cap * dim);
  std::vector<S> combined_scores(new_cap);
  std::copy_n(d_tmp_keys.h_data, table_size_after, combined_keys.begin());
  std::copy_n(d_tmp_values.h_data, table_size_after * dim, combined_values.begin());
  std::copy_n(d_tmp_scores.h_data, table_size_after, combined_scores.begin());
  std::copy_n(evict_buffer->keys_ptr(false), filtered_len,
              combined_keys.begin() + table_size_after);
  std::copy_n(evict_buffer->values_ptr(false), filtered_len * dim,
              combined_values.begin() + table_size_after * dim);
  std::copy_n(evict_buffer->scores_ptr(false), filtered_len,
              combined_scores.begin() + table_size_after);

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt1 = 0;
  size_t score_error_cnt2 = 0;

  for (int64_t i = static_cast<int64_t>(new_cap) - 1; i >= 0; --i) {
    const auto* vec = reinterpret_cast<const ValueArray<V, dim>*>(
        combined_values.data() + i * dim);
    values_map_after_insert[combined_keys[static_cast<size_t>(i)]] = *vec;
    scores_map_after_insert[combined_keys[static_cast<size_t>(i)]] =
        combined_scores[static_cast<size_t>(i)];
    if (static_cast<size_t>(i) >= (new_cap - filtered_len)) {
      if ((combined_scores[static_cast<size_t>(i)] >> 32) >=
          (global_epoch - 2)) {
        ++score_error_cnt1;
      }
    }
  }

  for (const auto& item : scores_map_current_batch) {
    const S score = scores_map_after_insert[item.first];
    const bool valid =
        ((score >> 32) == global_epoch) &&
        ((score & 0xFFFFFFFF) >= (0xFFFFFFFF & (nano_before_insert >> 20))) &&
        ((score & 0xFFFFFFFF) <= (0xFFFFFFFF & (nano_after_insert >> 20)));
    if (!valid) {
      ++score_error_cnt2;
    }
  }
  for (const auto& item : values_map_before_insert) {
    const auto after_it = values_map_after_insert.find(item.first);
    if (after_it == values_map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    for (size_t j = 0; j < dim; ++j) {
      if (item.second.data[j] != after_it->second.data[j]) {
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
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
  ASSERT_EQ(score_error_cnt1, 0);
  ASSERT_EQ(score_error_cnt2, 0);
}

void test_insert_and_evict_advanced_on_epochlru() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 128 * 1024;

  HashTableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  opt.num_of_buckets_per_alloc = 32;
  opt.dim = dim;
  opt.io_by_cpu = false;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kEpochLru>;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  auto table = std::make_unique<Table>();
  table->init(opt);

  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(B, dim, stream);
  evict_buffer.to_zeros(stream);

  KVMSBuffer<i64, f32, u64> data_buffer;
  data_buffer.reserve(B, dim, stream);

  for (unsigned int global_epoch = 1; global_epoch <= 64; ++global_epoch) {
    create_random_keys_advanced<i64, u64, f32>(
        dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), B, B * 16);
    data_buffer.sync_data(true, stream);

    CheckInsertAndEvictOnEpochLru<i64, f32, u64, Table>(
        table.get(), &data_buffer, &evict_buffer, B, stream, global_epoch);
  }

  ACL_CHECK(aclrtDestroyStream(stream));
}

template <typename K, typename V, typename S, typename Table>
void CheckInsertAndEvictOnEpochLfu(
    Table* table, KVMSBuffer<K, V, S>* data_buffer,
    KVMSBuffer<K, V, S>* evict_buffer,
    KVMSBuffer<K, V, S>* pre_data_buffer, size_t len, aclrtStream stream,
    unsigned int global_epoch) {
  std::map<K, ValueArray<V, dim>> values_map_before_insert;
  std::map<K, ValueArray<V, dim>> values_map_after_insert;
  std::unordered_map<K, S> scores_map_before_insert;
  std::map<K, S> scores_map_after_insert;
  std::map<K, S> scores_map_current_batch;
  std::map<K, S> scores_map_current_evict;

  for (size_t i = 0; i < len; ++i) {
    scores_map_current_batch[data_buffer->keys_ptr(false)[i]] =
        data_buffer->scores_ptr(false)[i];
  }

  const size_t table_size_before = table->size(stream);
  const size_t cap = table_size_before + len;

  HostAndDeviceBuffer<K> d_tmp_keys;
  HostAndDeviceBuffer<V> d_tmp_values;
  HostAndDeviceBuffer<S> d_tmp_scores;
  HostAndDeviceBuffer<bool> d_tmp_founds;
  HostAndDeviceBuffer<V> d_find_values;

  size_t table_size_verify0 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify0, stream);
  ASSERT_EQ(table_size_before, table_size_verify0);

  emplace_value_map(values_map_before_insert, d_tmp_keys.h_data,
                    d_tmp_values.h_data, table_size_before);
  emplace_value_map(values_map_before_insert, data_buffer->keys_ptr(false),
                    data_buffer->values_ptr(false), len);
  for (size_t i = 0; i < table_size_before; ++i) {
    scores_map_before_insert[d_tmp_keys.h_data[i]] = d_tmp_scores.h_data[i];
  }

  auto start = std::chrono::steady_clock::now();
  table->set_global_epoch(global_epoch);
  const size_t filtered_len = table->insert_and_evict(
      len, data_buffer->keys_ptr(), data_buffer->values_ptr(),
      data_buffer->scores_ptr(), evict_buffer->keys_ptr(),
      evict_buffer->values_ptr(), evict_buffer->scores_ptr(), stream);
  evict_buffer->sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  auto end = std::chrono::steady_clock::now();
  const auto diff =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  d_tmp_founds.alloc(len, stream);
  d_find_values.alloc(len * dim, stream);

  {
    d_tmp_founds.to_zeros(stream);
    table->find(len, pre_data_buffer->keys_ptr(), d_find_values.d_data,
                d_tmp_founds.d_data, pre_data_buffer->scores_ptr(), stream);
    d_tmp_founds.sync_data(false, stream);
    pre_data_buffer->scores.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t found_counter = 0;
    size_t old_epoch_counter = 0;
    size_t new_epoch_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++found_counter;
      }
      const S score = pre_data_buffer->scores_ptr(false)[i];
      const S cur_epoch = score >> 32;
      if (global_epoch == cur_epoch) {
        ++new_epoch_counter;
      }
      if (global_epoch - 1 == cur_epoch) {
        ++old_epoch_counter;
      }
    }
    ASSERT_EQ(len, new_epoch_counter + old_epoch_counter);
    std::cout << "old_epoch_counter:" << old_epoch_counter
              << ", new_epoch_counter:" << new_epoch_counter << std::endl
              << ", pre_data filtered_len:" << filtered_len
              << ", pre_data miss counter:" << len - found_counter << std::endl;
    ASSERT_EQ(len, found_counter);

    d_tmp_founds.to_zeros(stream);
    table->contains(len, pre_data_buffer->keys_ptr(), d_tmp_founds.d_data,
                    stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t contains_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++contains_counter;
      }
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  {
    d_tmp_founds.to_zeros(stream);
    table->find(len, data_buffer->keys_ptr(), d_find_values.d_data,
                d_tmp_founds.d_data, data_buffer->scores_ptr(), stream);
    d_tmp_founds.sync_data(false, stream);
    data_buffer->scores.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t found_counter = 0;
    size_t new_epoch_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      const S score = data_buffer->scores_ptr(false)[i];
      const S cur_epoch = score >> 32;
      if (d_tmp_founds.h_data[i]) {
        ++found_counter;
      }
      if (global_epoch == cur_epoch) {
        ++new_epoch_counter;
      }
    }
    ASSERT_EQ(len, new_epoch_counter);
    std::cout << "filtered_len:" << filtered_len
              << ", miss counter:" << len - found_counter << std::endl;
    ASSERT_EQ(len, found_counter);

    d_tmp_founds.to_zeros(stream);
    table->contains(len, data_buffer->keys_ptr(), d_tmp_founds.d_data, stream);
    d_tmp_founds.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    size_t contains_counter = 0;
    for (size_t i = 0; i < len; ++i) {
      if (d_tmp_founds.h_data[i]) {
        ++contains_counter;
      }
    }
    ASSERT_EQ(contains_counter, found_counter);
  }

  {
    std::unordered_set<K> unique_keys;
    for (size_t i = 0; i < len; ++i) {
      unique_keys.insert(data_buffer->keys_ptr(false)[i]);
      unique_keys.insert(pre_data_buffer->keys_ptr(false)[i]);
    }
    const float repeat_rate =
        static_cast<float>(len * 2.0 - unique_keys.size()) / (len * 1.0);
    std::cout << "repeat_rate:" << repeat_rate << std::endl;
  }

  for (size_t i = 0; i < filtered_len; ++i) {
    scores_map_current_evict[evict_buffer->keys_ptr(false)[i]] =
        evict_buffer->scores_ptr(false)[i];
  }

  const float dur = static_cast<float>(diff.count());
  const size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = 0;
  capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                         &table_size_verify1, stream);
  ASSERT_EQ(table_size_verify1, table_size_after);

  const size_t new_cap = table_size_after + filtered_len;
  std::vector<K> combined_keys(new_cap);
  std::vector<V> combined_values(new_cap * dim);
  std::vector<S> combined_scores(new_cap);
  std::copy_n(d_tmp_keys.h_data, table_size_after, combined_keys.begin());
  std::copy_n(d_tmp_values.h_data, table_size_after * dim, combined_values.begin());
  std::copy_n(d_tmp_scores.h_data, table_size_after, combined_scores.begin());
  std::copy_n(evict_buffer->keys_ptr(false), filtered_len,
              combined_keys.begin() + table_size_after);
  std::copy_n(evict_buffer->values_ptr(false), filtered_len * dim,
              combined_values.begin() + table_size_after * dim);
  std::copy_n(evict_buffer->scores_ptr(false), filtered_len,
              combined_scores.begin() + table_size_after);

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  size_t score_error_cnt1 = 0;
  size_t score_error_cnt2 = 0;

  for (int64_t i = static_cast<int64_t>(new_cap) - 1; i >= 0; --i) {
    const auto* vec = reinterpret_cast<const ValueArray<V, dim>*>(
        combined_values.data() + i * dim);
    values_map_after_insert[combined_keys[static_cast<size_t>(i)]] = *vec;
    scores_map_after_insert[combined_keys[static_cast<size_t>(i)]] =
        combined_scores[static_cast<size_t>(i)];
    if (static_cast<size_t>(i) >= (new_cap - filtered_len)) {
      if ((combined_scores[static_cast<size_t>(i)] >> 32) >=
          (global_epoch - 2)) {
        ++score_error_cnt1;
      }
    }
  }

  for (const auto& item : scores_map_current_batch) {
    const K key = item.first;
    const S score = item.second;
    const S current_score = scores_map_after_insert[key];
    S score_before_insert = 0;
    if (scores_map_before_insert.find(key) != scores_map_before_insert.end() &&
        scores_map_current_evict.find(key) == scores_map_current_evict.end()) {
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

  for (const auto& item : values_map_before_insert) {
    const auto after_it = values_map_after_insert.find(item.first);
    if (after_it == values_map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    for (size_t j = 0; j < dim; ++j) {
      if (item.second.data[j] != after_it->second.data[j]) {
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
            << ", while len: " << len << ", dur: " << dur << std::endl;

  ASSERT_EQ(key_miss_cnt, 0);
  ASSERT_EQ(value_diff_cnt, 0);
  ASSERT_EQ(score_error_cnt1, 0);
  ASSERT_EQ(score_error_cnt2, 0);
}

void test_insert_and_evict_advanced_on_epochlfu() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 128 * 1024;

  HashTableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  opt.num_of_buckets_per_alloc = 32;
  opt.dim = dim;
  opt.io_by_cpu = false;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kEpochLfu>;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  auto table = std::make_unique<Table>();
  table->init(opt);

  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(B, dim, stream);
  evict_buffer.to_zeros(stream);

  KVMSBuffer<i64, f32, u64> data_buffer;
  KVMSBuffer<i64, f32, u64> pre_data_buffer;
  data_buffer.reserve(B, dim, stream);
  pre_data_buffer.reserve(B, dim, stream);

  constexpr int freq_range = 100;
  constexpr float repeat_rate = 0.9f;
  for (unsigned int global_epoch = 1; global_epoch <= 64; ++global_epoch) {
    if (global_epoch <= 1) {
      create_random_keys_advanced<i64, u64, f32>(
          dim, data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
          data_buffer.values_ptr(false), B, B * 16, freq_range);
    } else {
      create_random_keys_advanced<i64, u64, f32>(
          dim, data_buffer.keys_ptr(false), pre_data_buffer.keys_ptr(false),
          data_buffer.scores_ptr(false), data_buffer.values_ptr(false), B,
          B * 16, freq_range, repeat_rate);
    }
    data_buffer.sync_data(true, stream);
    if (global_epoch <= 1) {
      pre_data_buffer.copy_from(data_buffer, stream);
    }

    CheckInsertAndEvictOnEpochLfu<i64, f32, u64, Table>(
        table.get(), &data_buffer, &evict_buffer, &pre_data_buffer, B, stream,
        global_epoch);

    pre_data_buffer.copy_from(data_buffer, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_insert_and_evict_advanced_on_customized() {
  const size_t U = 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 100000;

  HashTableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.max_bucket_size = 128;
  opt.num_of_buckets_per_alloc = 2;
  opt.dim = dim;
  opt.io_by_cpu = false;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kCustomized>;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  auto table = std::make_unique<Table>();
  table->init(opt);

  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(B, dim, stream);
  evict_buffer.to_zeros(stream);

  KVMSBuffer<i64, f32, u64> data_buffer;
  data_buffer.reserve(B, dim, stream);

  for (int i = 0; i < 32; ++i) {
    create_random_keys<i64, u64, f32, dim>(
        data_buffer.keys_ptr(false), data_buffer.scores_ptr(false),
        data_buffer.values_ptr(false), B, B * 16);
    data_buffer.sync_data(true, stream);

    CheckInsertAndEvict<i64, f32, u64, Table>(table.get(), &data_buffer,
                                              &evict_buffer, B, stream);
  }

  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_insert_and_evict_with_export_batch() {
  const size_t max_capacity = 4096;
  const size_t init_capacity = 2048;
  size_t offset = 0;
  const size_t uplimit = 1048576;
  const size_t len = 4096 + 13;

  HashTableOptions opt{};
  opt.max_capacity = max_capacity;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = uplimit * dim * sizeof(f32);
  opt.num_of_buckets_per_alloc = 16;
  opt.dim = dim;
  opt.io_by_cpu = false;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kLru>;
  using Vec = ValueArray<f32, dim>;

  std::map<i64, Vec> ref_map;
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  auto table = std::make_unique<Table>();
  table->init(opt);

  KVMSBuffer<i64, f32, u64> buffer;
  buffer.reserve(len, dim, stream);
  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(len, dim, stream);

  buffer.to_range(offset, 1, stream);
  const size_t n_evicted = table->insert_and_evict(
      len, buffer.keys_ptr(), buffer.values_ptr(), nullptr,
      evict_buffer.keys_ptr(), evict_buffer.values_ptr(), nullptr, stream);
  std::printf("Insert %zu keys and evict %zu\n", len, n_evicted);
  evict_buffer.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  for (size_t i = 0; i < n_evicted; ++i) {
    const auto* vec =
        reinterpret_cast<const Vec*>(evict_buffer.values_ptr(false) + i * dim);
    ref_map[evict_buffer.keys_ptr(false)[i]] = *vec;
  }

  offset = 0;
  size_t search_len = (table->capacity() >> 2);
  for (; offset < table->capacity(); offset += search_len) {
    if (offset + search_len > table->capacity()) {
      search_len = table->capacity() - offset;
    }
    const size_t n_exported = table->export_batch(search_len, offset,
                                                  buffer.keys_ptr(),
                                                  buffer.values_ptr(),
                                                  nullptr, stream);
    buffer.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (size_t i = 0; i < n_exported; ++i) {
      const auto* vec =
          reinterpret_cast<const Vec*>(buffer.values_ptr(false) + i * dim);
      for (size_t j = 0; j < dim; ++j) {
        ASSERT_EQ(buffer.keys_ptr(false)[i], vec->operator[](j));
      }
      ref_map[buffer.keys_ptr(false)[i]] = *vec;
    }
  }

  for (const auto& item : ref_map) {
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(static_cast<f32>(item.first), item.second.data[j]);
    }
  }

  ACL_CHECK(aclrtDestroyStream(stream));
}

template <typename K, typename V, typename S, typename Table>
void BatchCheckInsertAndEvict(Table* table, K* keys, V* values, S* scores,
                              K* evicted_keys, V* evicted_values,
                              S* evicted_scores, size_t len,
                              std::atomic<int>* step, size_t total_step,
                              aclrtStream stream, bool if_check = true) {
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
  HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                  "aclrtSetDevice failed");
  int current_step = 0;
  while ((current_step = step->load()) < static_cast<int>(total_step)) {
    const size_t table_size_before = table->size(stream);
    const size_t cap = table_size_before + len;
    size_t key_miss_cnt = 0;
    size_t value_diff_cnt = 0;
    size_t table_size_after = 0;

    std::map<i64, ValueArray<f32, dim>> map_before_insert;
    std::map<i64, ValueArray<f32, dim>> map_after_insert;
    HostAndDeviceBuffer<K> d_tmp_keys;
    HostAndDeviceBuffer<V> d_tmp_values;
    HostAndDeviceBuffer<S> d_tmp_scores;

    if (if_check) {
      size_t table_size_verify0 = 0;
      capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                             &table_size_verify0, stream);
      ASSERT_EQ(table_size_before, table_size_verify0);

      emplace_value_map(map_before_insert, d_tmp_keys.h_data, d_tmp_values.h_data,
                        table_size_before);

      std::vector<K> batch_keys(len);
      std::vector<V> batch_values(len * dim);
      std::vector<S> batch_scores(len);
      ACL_CHECK(aclrtMemcpyAsync(batch_keys.data(), len * sizeof(K),
                                 keys + len * current_step, len * sizeof(K),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
      ACL_CHECK(aclrtMemcpyAsync(batch_values.data(), len * dim * sizeof(V),
                                 values + len * current_step * dim,
                                 len * dim * sizeof(V),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
      ACL_CHECK(aclrtMemcpyAsync(batch_scores.data(), len * sizeof(S),
                                 scores + len * current_step, len * sizeof(S),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
      ACL_CHECK(aclrtSynchronizeStream(stream));
      emplace_value_map(map_before_insert, batch_keys.data(), batch_values.data(),
                        len);
    }

    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto start = std::chrono::steady_clock::now();
    const size_t filtered_len = table->insert_and_evict(
        len, keys + len * current_step, values + len * current_step * dim,
        nullptr, evicted_keys, evicted_values, evicted_scores, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto end = std::chrono::steady_clock::now();
    const auto diff =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    if (if_check) {
      table_size_after = table->size(stream);
      size_t table_size_verify1 = 0;
      capture_table_snapshot(table, cap, d_tmp_keys, d_tmp_values, d_tmp_scores,
                             &table_size_verify1, stream);
      ASSERT_EQ(table_size_verify1, table_size_after);

      const size_t new_cap = table_size_after + filtered_len;
      std::vector<K> combined_keys(new_cap);
      std::vector<V> combined_values(new_cap * dim);
      std::copy_n(d_tmp_keys.h_data, table_size_after, combined_keys.begin());
      std::copy_n(d_tmp_values.h_data, table_size_after * dim,
                  combined_values.begin());

      std::vector<K> host_evicted_keys(filtered_len);
      std::vector<V> host_evicted_values(filtered_len * dim);
      std::vector<S> host_evicted_scores(filtered_len);
      ACL_CHECK(aclrtMemcpyAsync(host_evicted_keys.data(), filtered_len * sizeof(K),
                                 evicted_keys, filtered_len * sizeof(K),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
      ACL_CHECK(aclrtMemcpyAsync(host_evicted_values.data(),
                                 filtered_len * dim * sizeof(V),
                                 evicted_values,
                                 filtered_len * dim * sizeof(V),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
      ACL_CHECK(aclrtMemcpyAsync(host_evicted_scores.data(),
                                 filtered_len * sizeof(S), evicted_scores,
                                 filtered_len * sizeof(S),
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream));
      ACL_CHECK(aclrtSynchronizeStream(stream));
      std::copy_n(host_evicted_keys.data(), filtered_len,
                  combined_keys.begin() + table_size_after);
      std::copy_n(host_evicted_values.data(), filtered_len * dim,
                  combined_values.begin() + table_size_after * dim);

      for (int64_t i = static_cast<int64_t>(new_cap) - 1; i >= 0; --i) {
        const auto* vec = reinterpret_cast<const ValueArray<V, dim>*>(
            combined_values.data() + i * dim);
        map_after_insert[combined_keys[static_cast<size_t>(i)]] = *vec;
      }

      for (const auto& item : map_before_insert) {
        const auto after_it = map_after_insert.find(item.first);
        if (after_it == map_after_insert.end()) {
          ++key_miss_cnt;
          continue;
        }
        for (size_t j = 0; j < dim; ++j) {
          if (item.second.data[j] != after_it->second.data[j]) {
            ++value_diff_cnt;
            break;
          }
        }
      }
      ASSERT_EQ(key_miss_cnt, 0);
      ASSERT_EQ(value_diff_cnt, 0);
    }

    std::cout << "Check insert behavior got step: " << step->load()
              << ",\tduration: " << diff.count()
              << ",\twhile value_diff_cnt: " << value_diff_cnt
              << ", while table_size_before: " << table_size_before
              << ", while table_size_after: " << table_size_after
              << ", while len: " << len << std::endl;

    step->fetch_add(1);
  }
}

template <typename K, typename V, typename S, typename Table>
void BatchCheckFind(Table* table, K* keys, V* values, S* scores, size_t len,
                    std::atomic<int>* step, size_t total_step,
                    size_t find_interval, aclrtStream stream,
                    bool if_check = true) {
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
  HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                  "aclrtSetDevice failed");
  int find_step = 0;
  const size_t cap = len * find_interval;

  HostAndDeviceBuffer<K> d_tmp_keys;
  HostAndDeviceBuffer<V> d_tmp_values;
  HostAndDeviceBuffer<S> d_tmp_scores;
  HostAndDeviceBuffer<bool> d_tmp_founds;
  d_tmp_keys.alloc(cap, stream);
  d_tmp_values.alloc(cap * dim, stream);
  d_tmp_scores.alloc(cap, stream);
  d_tmp_founds.alloc(cap, stream);

  while (step->load() < static_cast<int>(total_step)) {
    while (find_step >= (step->load() / static_cast<int>(find_interval))) {
    }

    size_t found_num = 0;
    size_t value_diff_cnt = 0;
    d_tmp_keys.to_zeros(stream);
    d_tmp_values.to_zeros(stream);
    d_tmp_scores.to_zeros(stream);
    d_tmp_founds.to_zeros(stream);

    ACL_CHECK(aclrtMemcpyAsync(d_tmp_keys.d_data, cap * sizeof(K),
                               keys + cap * find_step, cap * sizeof(K),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    auto start = std::chrono::steady_clock::now();
    table->find(cap, d_tmp_keys.d_data, d_tmp_values.d_data, d_tmp_founds.d_data,
                d_tmp_scores.d_data, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto end = std::chrono::steady_clock::now();
    const auto diff =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    if (if_check) {
      d_tmp_keys.sync_data(false, stream);
      d_tmp_values.sync_data(false, stream);
      d_tmp_scores.sync_data(false, stream);
      d_tmp_founds.sync_data(false, stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));

      for (size_t i = 0; i < cap; ++i) {
        if (d_tmp_founds.h_data[i]) {
          for (size_t j = 0; j < dim; ++j) {
            if (d_tmp_values.h_data[i * dim + j] !=
                static_cast<f32>(d_tmp_keys.h_data[i] * 0.00001)) {
              ++value_diff_cnt;
            }
          }
          ++found_num;
        }
      }
      ASSERT_EQ(value_diff_cnt, 0);

      d_tmp_founds.to_zeros(stream);
      table->contains(cap, keys, d_tmp_founds.d_data, stream);
      d_tmp_founds.sync_data(false, stream);
      ACL_CHECK(aclrtSynchronizeStream(stream));
      size_t contains_num = 0;
      for (size_t i = 0; i < cap; ++i) {
        if (d_tmp_founds.h_data[i]) {
          ++contains_num;
        }
      }
      ASSERT_EQ(contains_num, found_num);
    }

    std::cout << std::endl
              << "\nCheck find behavior got step: " << find_step
              << ",\tduration: " << diff.count()
              << ",\twhile value_diff_cnt: " << value_diff_cnt
              << ", while cap: " << cap << std::endl
              << std::endl;
    ASSERT_EQ(value_diff_cnt, 0);
    ++find_step;
  }
}

void test_insert_and_evict_run_with_batch_find() {
  const size_t U = 16 * 1024 * 1024;
  const size_t init_capacity = U;
  const size_t B = 256 * 1024;
  constexpr size_t batch_num = 256;
  constexpr size_t find_interval = 8;

  const bool if_check = false;

  std::thread insert_and_evict_thread;
  std::thread find_thread;
  std::atomic<int> step{0};

  HashTableOptions opt{};
  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.num_of_buckets_per_alloc = 128;
  opt.dim = dim;
  opt.io_by_cpu = false;
  opt.api_lock = true;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kLru>;

  aclrtStream insert_stream = nullptr;
  aclrtStream find_stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&insert_stream));
  ACL_CHECK(aclrtCreateStream(&find_stream));

  auto table = std::make_unique<Table>();
  table->init(opt);

  KVMSBuffer<i64, f32, u64> global_buffer;
  global_buffer.reserve(B * batch_num, dim, insert_stream);

  KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.reserve(B, dim, insert_stream);
  evict_buffer.to_zeros(insert_stream);

  for (size_t i = 0; i < batch_num; ++i) {
    create_random_keys<i64, u64, f32, dim>(
        global_buffer.keys_ptr(false) + B * i,
        global_buffer.scores_ptr(false) + B * i,
        global_buffer.values_ptr(false) + B * i * dim, B);
  }
  global_buffer.sync_data(true, insert_stream);
  ACL_CHECK(aclrtSynchronizeStream(insert_stream));

  auto insert_and_evict_func = [&table, &global_buffer, &evict_buffer, &B,
                                &step, &batch_num, &insert_stream, if_check]() {
    BatchCheckInsertAndEvict<i64, f32, u64, Table>(
        table.get(), global_buffer.keys_ptr(), global_buffer.values_ptr(),
        global_buffer.scores_ptr(), evict_buffer.keys_ptr(),
        evict_buffer.values_ptr(), evict_buffer.scores_ptr(), B, &step,
        batch_num, insert_stream, if_check);
  };

  auto find_func = [&table, &global_buffer, &B, &step, &batch_num,
                    &find_interval, &find_stream, if_check]() {
    BatchCheckFind<i64, f32, u64, Table>(
        table.get(), global_buffer.keys_ptr(), global_buffer.values_ptr(),
        global_buffer.scores_ptr(), B, &step, batch_num, find_interval,
        find_stream, if_check);
  };

  find_thread = std::thread(find_func);
  insert_and_evict_thread = std::thread(insert_and_evict_func);
  find_thread.join();
  insert_and_evict_thread.join();

  ACL_CHECK(aclrtDestroyStream(insert_stream));
  ACL_CHECK(aclrtDestroyStream(find_stream));
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_basic) {
  test_insert_and_evict_basic();
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_advanced_on_lru) {
  test_insert_and_evict_advanced_on_lru();
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_advanced_on_lfu) {
  test_insert_and_evict_advanced_on_lfu();
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_advanced_on_epochlru) {
  test_insert_and_evict_advanced_on_epochlru();
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_advanced_on_epochlfu) {
  test_insert_and_evict_advanced_on_epochlfu();
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_advanced_on_customized) {
  test_insert_and_evict_advanced_on_customized();
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_with_export_batch) {
  test_insert_and_evict_with_export_batch();
}

TEST_F(InsertAndEvictTest, test_insert_and_evict_run_with_batch_find) {
  test_insert_and_evict_run_with_batch_find();
}

}  // namespace
