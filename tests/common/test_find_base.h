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

#pragma once

#include <gtest/gtest.h>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

// 基类：提供通用的表操作和设备内存管理，供 find / find_miss / contains 等复用
class FindTestBase : public ::testing::Test {
 public:
  static constexpr size_t DEFAULT_DIM = 8;
  static constexpr size_t DEFAULT_INIT_CAPACITY = 128UL * 1024;
  static constexpr size_t DEFAULT_HBM_FOR_VALUES = 1UL << 30;

  void SetUp() override {
    init_env();

    size_t total_mem = 0;
    size_t free_mem = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem), ACL_ERROR_NONE);
    ASSERT_GT(free_mem, DEFAULT_HBM_FOR_VALUES);
  }

  template<typename TableType>
  void InitTable(TableType& table, size_t dim, size_t init_capacity,
                 size_t hbm_for_values) {
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    table.init(options);
  }

  // 插入连续键 [start, start+key_num) 到表中，返回host侧的keys/values/scores
  template<typename TableType, size_t DIM>
  void InsertContinuousKeys(TableType& table, size_t key_num,
                            vector<K>& host_keys, vector<V>& host_values,
                            vector<S>& host_scores, bool with_scores,
                            aclrtStream stream, K start = 1) {
    host_keys.resize(key_num);
    host_values.resize(key_num * DIM);
    host_scores.resize(with_scores ? key_num : 0);

    create_continuous_keys<K, S, V, DIM>(
        host_keys.data(), with_scores ? host_scores.data() : nullptr,
        host_values.data(), key_num, start);

    K* device_keys = nullptr;
    V* device_values = nullptr;
    S* device_scores = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                          key_num * DIM * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                          key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_values, key_num * DIM * sizeof(V),
                          host_values.data(), key_num * DIM * sizeof(V),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    if (with_scores) {
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                            key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMemcpy(device_scores, key_num * sizeof(S),
                            host_scores.data(), key_num * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
    }

    table.insert_or_assign(key_num, device_keys, device_values,
                           device_scores, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

    ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
    if (device_scores) {
      ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
    }
  }
};
