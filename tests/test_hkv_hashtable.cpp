/* *
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
using TableOptions = npu::hkv::HashTableOptions;
using BaseAllocator = npu::hkv::BaseAllocator;
using MemoryType = npu::hkv::MemoryType;
using EvictStrategy = npu::hkv::EvictStrategy;

void test_basic(size_t max_hbm_for_vectors) {
  init_env();

  constexpr uint64_t BUCKET_MAX_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL - (128 + 1);
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = npu::hkv::GB(max_hbm_for_vectors);
  options.reserved_key_start_bit = 2;
  options.num_of_buckets_per_alloc = 32;

  using Table = npu::hkv::HashTable<K, V, S, EvictStrategy::kCustomized>;

  NPU_CHECK(aclrtMallocHost((void**)&h_keys, KEY_NUM * sizeof(K)));
  NPU_CHECK(aclrtMallocHost((void**)&h_scores, KEY_NUM * sizeof(S)));
  NPU_CHECK(
      aclrtMallocHost((void**)&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  NPU_CHECK(aclrtMallocHost((void**)&h_found, KEY_NUM * sizeof(bool)));

  NPU_CHECK(aclrtMemset(h_vectors, KEY_NUM * sizeof(V) * options.dim, 0,
                        KEY_NUM * sizeof(V) * options.dim));

  test_util::create_random_keys<K, S, V, DIM>(h_keys, h_scores, h_vectors,
                                              KEY_NUM);
  K* d_keys;
  S* d_scores = nullptr;
  V* d_vectors;
  V* d_new_vectors;
  V** d_vectors_ptr = nullptr;
  bool* d_found;
  size_t dump_counter = 0;

  NPU_CHECK(aclrtMalloc((void**)&d_keys, KEY_NUM * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_scores, KEY_NUM * sizeof(S),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_new_vectors,
                        KEY_NUM * sizeof(V) * options.dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_found, KEY_NUM * sizeof(bool),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr, KEY_NUM * sizeof(V*),
                        ACL_MEM_MALLOC_HUGE_FIRST));

  NPU_CHECK(aclrtMemcpy(d_keys, KEY_NUM * sizeof(K), h_keys,
                        KEY_NUM * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_scores, KEY_NUM * sizeof(K), h_scores,
                        KEY_NUM * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_vectors, KEY_NUM * sizeof(V) * options.dim, h_vectors,
                        KEY_NUM * sizeof(V) * options.dim,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  NPU_CHECK(
      aclrtMemset(d_found, KEY_NUM * sizeof(bool), 0, KEY_NUM * sizeof(bool)));

  aclrtStream stream = nullptr;
  NPU_CHECK(aclrtCreateStream(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    ASSERT_EQ(table->bucket_count(),
              524287);  // 1 + (INIT_CAPACITY / options.bucket_max_size)
    total_size = table->size(stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, 0);
    ASSERT_EQ(table->empty(stream), true);
    ASSERT_EQ(table->dim(), DIM);
    ASSERT_EQ(table->max_bucket_size(), BUCKET_MAX_SIZE);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_scores, stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));

    total_size = table->size(stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    table->find_or_insert(KEY_NUM, d_keys, d_vectors_ptr, d_found, d_scores,
                          stream);
    total_size = table->size(stream);
    test_util::read_from_ptr(d_vectors_ptr, d_vectors, options.dim, KEY_NUM,
                             stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    int found_num = 0;
    NPU_CHECK(aclrtMemcpy(h_found, KEY_NUM * sizeof(bool), d_found,
                          KEY_NUM * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtMemcpy(h_scores, KEY_NUM * sizeof(S), d_scores,
                          KEY_NUM * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtMemcpy(h_vectors, KEY_NUM * sizeof(V) * options.dim,
                          d_vectors, KEY_NUM * sizeof(V) * options.dim,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
      ASSERT_EQ(h_scores[i], h_keys[i]);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  static_cast<float>(h_keys[i] * 0.00001));
      }
    }
  }

  NPU_CHECK(aclrtDestroyStream(stream));

  NPU_CHECK(aclrtFreeHost(h_keys));
  NPU_CHECK(aclrtFreeHost(h_scores));
  NPU_CHECK(aclrtFreeHost(h_vectors));
  NPU_CHECK(aclrtFreeHost(h_found));

  NPU_CHECK(aclrtFree(d_keys));
  NPU_CHECK(aclrtFree(d_scores));
  NPU_CHECK(aclrtFree(d_vectors));
  NPU_CHECK(aclrtFree(d_new_vectors));
  NPU_CHECK(aclrtFree(d_found));
  NPU_CHECK(aclrtFree(d_vectors_ptr));

  NPU_CHECK(aclrtSynchronizeDevice());
  NpuCheckError();
}

TEST(HkvHashTableTest, test_basic) {
  test_basic(32);
}
