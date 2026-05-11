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
#include <cstdio>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace community_test_util;

namespace {

constexpr size_t dim = 64;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using TableOptions = HashTableOptions;

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

/*
 * test_dynamic_max_capcity_table creates a table in small
 * capacity and insert random kv pairs until its load_factor
 * became 1.0. Then expand the max_capacity. Keep inserting until
 * the load factor growth to 1.0 again.
 */
void test_dynamic_max_capcity_table() {
  const size_t len = 10000llu;
  size_t max_capacity = 1 << 14;
  const size_t init_capacity = 1 << 12;
  size_t offset = 0;
  const size_t uplimit = 1 << 20;
  const float load_factor_threshold = 0.98f;

  TableOptions opt;
  opt.max_capacity = max_capacity;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = uplimit * dim * sizeof(f32);
  opt.dim = dim;
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

  size_t total_len = 0;
  while (true) {
    buffer.to_range(offset, 1, stream);
    const size_t n_evicted = table->insert_and_evict(
        len, buffer.keys_ptr(), buffer.values_ptr(), nullptr,
        evict_buffer.keys_ptr(), evict_buffer.values_ptr(), nullptr, stream);
    std::printf("Insert %zu keys and evict %zu\n", len, n_evicted);
    offset += len;
    total_len += len;

    evict_buffer.sync_data(false, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (size_t i = 0; i < n_evicted; ++i) {
      const auto* vec =
          reinterpret_cast<const Vec*>(evict_buffer.values_ptr(false) +
                                       i * dim);
      ref_map[evict_buffer.keys_ptr(false)[i]] = *vec;
    }

    if (table->load_factor(stream) >= load_factor_threshold) {
      ASSERT_GE(table->size(stream),
                static_cast<size_t>(static_cast<float>(max_capacity) *
                                    load_factor_threshold));
      max_capacity *= 2;
      if (max_capacity > uplimit) {
        break;
      }
      // What we need.
      std::printf("----> check change max_capacity from %zu to %zu\n",
                  table->capacity(), max_capacity);
      table->set_max_capacity(max_capacity);
      table->reserve(max_capacity, stream);
      ASSERT_EQ(max_capacity, table->capacity());
      ASSERT_LE(table->load_factor(stream), 0.5f);
    }

    if (total_len > uplimit * 2) {
      throw std::runtime_error("Traverse too much keys but not finish test.");
    }
  }

  offset = 0;
  for (; offset < table->capacity(); offset += len) {
    size_t search_len = len;
    if (offset + search_len > table->capacity()) {
      search_len = table->capacity() - offset;
    }
    const size_t n_exported =
        table->export_batch(search_len, offset, buffer.keys_ptr(),
                            buffer.values_ptr(), nullptr, stream);
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

  std::printf("---> uplimit: %zu\n", uplimit);
  std::printf("---> table size: %zu\n", table->size(stream));
  std::printf("---> table cap: %zu\n", table->capacity());
  std::printf("---> cpu table size: %zu\n", ref_map.size());
  for (const auto& it : ref_map) {
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(static_cast<f32>(it.first), it.second.data[j]);
    }
  }
  ASSERT_EQ(table->capacity() * 2, max_capacity);
  ASSERT_GE(static_cast<float>(ref_map.size()),
            static_cast<float>(table->capacity()) * load_factor_threshold);

  ACL_CHECK(aclrtDestroyStream(stream));
}

}  // namespace

TEST(MerlinHashTableTest, test_dynamic_max_capcity_table) {
  init_env();
  test_dynamic_max_capcity_table();
}
