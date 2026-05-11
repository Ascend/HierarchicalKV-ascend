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
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

constexpr uint64_t DIM = 64;
using K = int64_t;
using S = uint64_t;
using V = float;
using TableOptions = npu::hkv::HashTableOptions;

using namespace npu::hkv;
using namespace community_test_util;

namespace {

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

struct StreamGuard {
  aclrtStream stream = nullptr;

  ~StreamGuard() {
    if (stream != nullptr) {
      aclrtDestroyStream(stream);
    }
  }

  void create() { ACL_CHECK(aclrtCreateStream(&stream)); }
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

  void copy_to_host(std::vector<T>* host, size_t n, aclrtStream stream) const {
    ASSERT_LE(n, count);
    host->assign(n, T{});
    ACL_CHECK(aclrtMemcpyAsync(host->data(), sizeof(T) * n, ptr, sizeof(T) * n,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }
};

template <typename Table>
void tables_equal(Table* a, Table* b, bool check_score, aclrtStream stream,
                  bool* equal) {
  ASSERT_NE(equal, nullptr);
  *equal = false;

  size_t size = a->size(stream);
  if (size != b->size(stream)) {
    return;
  }

  if (a->dim() != b->dim()) {
    return;
  }

  DeviceArray<K> d_keys;
  DeviceArray<V> d_vectors;
  DeviceArray<S> d_scores;
  DeviceArray<bool> d_founds_in_b;
  DeviceArray<V> d_vectors_in_b;
  DeviceArray<S> d_scores_in_b;
  d_keys.alloc(size);
  d_vectors.alloc(size * a->dim());
  d_scores.alloc(size);
  d_founds_in_b.alloc(size);
  d_vectors_in_b.alloc(size * a->dim());
  d_scores_in_b.alloc(size);

  const size_t exported =
      a->export_batch(a->capacity(), 0, d_keys.get(), d_vectors.get(),
                      d_scores.get(), stream);
  ASSERT_EQ(exported, size);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  b->find(size, d_keys.get(), d_vectors_in_b.get(), d_founds_in_b.get(),
          d_scores_in_b.get(), stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::vector<uint8_t> h_found(size);
  ACL_CHECK(aclrtMemcpyAsync(h_found.data(), sizeof(uint8_t) * size,
                             d_founds_in_b.get(), sizeof(bool) * size,
                             ACL_MEMCPY_DEVICE_TO_HOST, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
  for (size_t i = 0; i < size; ++i) {
    if (!h_found[i]) {
      return;
    }
  }

  if (check_score) {
    std::vector<S> h_scores;
    std::vector<S> h_scores_in_b;
    d_scores.copy_to_host(&h_scores, size, stream);
    d_scores_in_b.copy_to_host(&h_scores_in_b, size, stream);
    if (h_scores != h_scores_in_b) {
      return;
    }
  }

  std::vector<V> h_vectors;
  std::vector<V> h_vectors_in_b;
  d_vectors.copy_to_host(&h_vectors, size * a->dim(), stream);
  d_vectors_in_b.copy_to_host(&h_vectors_in_b, size * a->dim(), stream);
  if (h_vectors != h_vectors_in_b) {
    return;
  }

  *equal = true;
}

template <typename Table>
void test_save_to_file() {
  std::string prefix = "checkpoint";
  size_t keynum = 1 * 1024 * 1024;
  size_t capacity = 2 * 1024 * 1024;
  size_t buffer_size = 1024 * 1024;
  init_env();
  StreamGuard guard;
  guard.create();

  HostAndDeviceBuffer<K> keys;
  HostAndDeviceBuffer<V> vectors;
  HostAndDeviceBuffer<S> scores;
  keys.alloc(keynum, guard.stream);
  vectors.alloc(keynum * DIM, guard.stream);
  scores.alloc(keynum, guard.stream);
  create_random_keys<K, S>(keys.h_data, scores.h_data,
                           static_cast<int>(keynum));
  keys.sync_data(true, guard.stream);
  scores.sync_data(true, guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  printf("Pass create random keys.\n");

  printf("Create buffers.\n");

  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.dim = DIM;

  std::unique_ptr<Table> table_0 = std::make_unique<Table>();
  std::unique_ptr<Table> table_1 = std::make_unique<Table>();
  table_0->init(options);
  table_1->init(options);
  printf("Init tables.\n");

  S global_epoch = 101;
  S* temp_score = (Table::evict_strategy == EvictStrategy::kLru ||
                   Table::evict_strategy == EvictStrategy::kEpochLru)
                      ? nullptr
                      : scores.d_data;
  table_0->set_global_epoch(global_epoch);
  table_0->insert_or_assign(keynum, keys.d_data, vectors.d_data, temp_score,
                            guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
  printf("Fill table_0.\n");

  LocalKVFile<K, V, S> file;
  std::string keys_path = prefix + ".keys";
  std::string values_path = prefix + ".values";
  std::string scores_path = prefix + ".scores";
  ASSERT_TRUE(file.open(keys_path, values_path, scores_path, "wb"));
  table_0->save(&file, buffer_size, guard.stream);
  file.close();
  printf("table_0 saves.\n");

  ASSERT_TRUE(file.open(keys_path, values_path, scores_path, "rb"));
  table_1->load(&file, buffer_size, guard.stream);
  file.close();
  printf("table_1 loads.\n");

  bool check_score = !(Table::evict_strategy == EvictStrategy::kLru ||
                       Table::evict_strategy == EvictStrategy::kEpochLru);
  bool equal = false;
  tables_equal<Table>(table_0.get(), table_1.get(), check_score, guard.stream,
                      &equal);
  ASSERT_TRUE(equal);
  printf("table_0 and table_1 are equal.\n");

  keys.free(guard.stream);
  vectors.free(guard.stream);
  scores.free(guard.stream);
  ACL_CHECK(aclrtSynchronizeStream(guard.stream));
}

}  // namespace

TEST(SaveAndLoadTest, test_save_and_load_on_lru) {
  test_save_to_file<HashTable<K, V, S, EvictStrategy::kLru>>();
}
TEST(SaveAndLoadTest, test_save_and_load_on_lfu) {
  test_save_to_file<HashTable<K, V, S, EvictStrategy::kLfu>>();
}
TEST(SaveAndLoadTest, test_save_and_load_on_epochlru) {
  test_save_to_file<HashTable<K, V, S, EvictStrategy::kEpochLru>>();
}
TEST(SaveAndLoadTest, test_save_and_load_on_epochlfu) {
  test_save_to_file<HashTable<K, V, S, EvictStrategy::kEpochLfu>>();
}
TEST(SaveAndLoadTest, test_save_and_load_on_customized) {
  test_save_to_file<HashTable<K, V, S, EvictStrategy::kCustomized>>();
}
