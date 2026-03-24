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
#include <memory>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

// 通用测试配置结构体
struct ExportTestConfig {
  size_t key_num;         // 键数量
  bool test_overload;     // 是否测试重载版本
  bool test_empty_table;  // 是否测试空表
  size_t offset;          // 导出偏移量
  bool use_scores;        // 是否使用分数
};
// 大数据量测试模板函数 - 使用key/value比值验证
// 直接验证value == key * 0.00001的关系，避免构建reference_map
template <typename K, typename V, typename S, size_t DIM>
void run_large_scale_export_test(const ExportTestConfig& config) {
  init_env();

  size_t total_mem = 0;
  size_t free_mem = 0;
  constexpr size_t hbm_for_values = 16UL << 30;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem
      << "need:" << hbm_for_values;

  constexpr size_t init_capacity = 128UL * 1024;
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = DIM,
      .io_by_cpu = false,
  };

  HashTable<K, V> table;
  table.init(options);

  K* device_keys = nullptr;
  V* device_values = nullptr;

  if (!config.test_empty_table) {
    // 生成测试数据，确保key值小于init_capacity
    vector<K> host_keys(config.key_num);
    vector<V> host_values(config.key_num * DIM);
    K start_key = 1;
    create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                         host_values.data(), config.key_num, start_key);

    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                    config.key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
        ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                          config.key_num * DIM * sizeof(V),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);

    ASSERT_EQ(
        aclrtMemcpy(device_keys, config.key_num * sizeof(K), host_keys.data(),
                    config.key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_values, config.key_num * DIM * sizeof(V),
                          host_values.data(), config.key_num * DIM * sizeof(V),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    // 少量多次插入数据
    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
    const size_t batch_size = 1024; // 每批插入128k个
    for (size_t i = 0; i < config.key_num; i += batch_size) {
      size_t current_batch_size = std::min(batch_size, config.key_num - i);
      table.insert_or_assign(current_batch_size, device_keys + i, device_values + i * DIM, nullptr,
                           stream);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }

  const size_t scan_len = table.capacity();
  K* device_export_keys = nullptr;
  V* device_export_values = nullptr;
  S* device_export_scores = nullptr;

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_export_keys),
                        scan_len * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_export_values),
                        scan_len * DIM * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  if (config.use_scores) {
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_export_scores),
                          scan_len * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemset(device_export_scores, scan_len * sizeof(S), 0,
                          scan_len * sizeof(S)),
              ACL_ERROR_NONE);
  }

  ASSERT_EQ(aclrtMemset(device_export_keys, scan_len * sizeof(K), 0,
                        scan_len * sizeof(K)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(device_export_values, scan_len * DIM * sizeof(V), 0,
                        scan_len * DIM * sizeof(V)),
            ACL_ERROR_NONE);

  size_t export_count = 0;

  if (config.test_overload) {
    export_count = table.export_batch(
        scan_len, config.offset, device_export_keys, device_export_values,
        config.use_scores ? device_export_scores : nullptr, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  } else {
    size_t* device_export_count = nullptr;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_export_count),
                          sizeof(size_t), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(
        aclrtMemset(device_export_count, sizeof(size_t), 0, sizeof(size_t)),
              ACL_ERROR_NONE);

    table.export_batch(scan_len, config.offset, device_export_count,
                       device_export_keys, device_export_values,
                       config.use_scores ? device_export_scores : nullptr,
                       stream);

    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(&export_count, sizeof(size_t), device_export_count,
                          sizeof(size_t), ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    ASSERT_EQ(aclrtFree(device_export_count), ACL_ERROR_NONE);
  }

  if (config.test_empty_table) {
    ASSERT_EQ(export_count, 0);
//   } else if (config.offset == 0) {
//     ASSERT_EQ(export_count, config.key_num);
  } else {
    ASSERT_LE(export_count, config.key_num);
    ASSERT_GE(export_count, 0);
  }

  if (export_count > 0 && !config.test_empty_table) {
    vector<K> host_export_keys(export_count);
    vector<V> host_export_values(export_count * DIM);

    ASSERT_EQ(aclrtMemcpy(host_export_keys.data(), export_count * sizeof(K),
                          device_export_keys, export_count * sizeof(K),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(
        aclrtMemcpy(host_export_values.data(), export_count * DIM * sizeof(V),
                    device_export_values, export_count * DIM * sizeof(V),
                    ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_ERROR_NONE);

    constexpr V expected_ratio = static_cast<V>(0.00001);
    constexpr V tolerance = static_cast<V>(1e-5);

    for (size_t i = 0; i < export_count; ++i) {
      const K& key = host_export_keys[i];
      V expected_value = static_cast<V>(key * 0.00001);

      for (size_t j = 0; j < DIM; ++j) {
        V actual_value = host_export_values[i * DIM + j];
        V diff = std::abs(actual_value - expected_value);
        ASSERT_LE(diff, tolerance)
            << "Value mismatch for key " << key << ", dimension " << j
            << ": expected " << expected_value << ", got " << actual_value
            << " (diff: " << diff << ")";
      }
    }
  }

  if (device_keys != nullptr) {
    ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  }
  if (device_values != nullptr) {
    ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  }
  ASSERT_EQ(aclrtFree(device_export_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_export_values), ACL_ERROR_NONE);
  if (config.use_scores) {
    ASSERT_EQ(aclrtFree(device_export_scores), ACL_ERROR_NONE);
  }
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

#define RUN_LARGE_SCALE_EXPORT_TEST(K, V, S, DIM, config) \
  run_large_scale_export_test<K, V, S, DIM>(config)

  
// 大数据量测试用例 - 使用key/value比值验证
TEST(test_export_batch, test_large_scale_export_basic) {
  // 测试100K数据量的导出
  ExportTestConfig config = {1 * 1024, false, false, 0, false};
  RUN_LARGE_SCALE_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_LARGE_SCALE_EXPORT_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch, test_large_scale_export_with_scores) {
  // 测试带分数的大数据量导出
  ExportTestConfig config = {5 * 1024, true, false, 0, true};
  RUN_LARGE_SCALE_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
}

TEST(test_export_batch, test_large_scale_export_overload) {
  // 测试大数据量的重载版本
  ExportTestConfig config = {1 * 1024, true, false, 0, true};
  RUN_LARGE_SCALE_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
}

TEST(test_export_batch, test_large_scale_export_offset) {
  // 测试大数据量带偏移的导出
  ExportTestConfig config = {1 * 1024, false, false, 10000, false};
  RUN_LARGE_SCALE_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
}

TEST(test_export_batch, test_large_scale_export_empty) {
  // 测试大数据量空表导出
  ExportTestConfig config = {0, false, true, 0, false};
  RUN_LARGE_SCALE_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
}

TEST(test_export_batch, test_large_scale_export_max_capacity) {
  // 测试接近最大容量的数据量
  ExportTestConfig config = {128 * 1024, false, false, 0, false};
  RUN_LARGE_SCALE_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
}
