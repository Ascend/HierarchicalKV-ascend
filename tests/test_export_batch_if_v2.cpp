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
#include "test_device_data.h"
#include "types.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

enum class FunctorVersion { V1, V2, V3, V4 };

// 通用测试配置结构体
struct ExportTestConfig {
  size_t key_num;         // 键数量
  bool test_empty_table;  // 是否测试空表
  size_t offset;          // 导出偏移量
  bool use_scores;        // 是否使用分数
  FunctorVersion functor_version;  // 导出函数版本
  bool use_fast_mode;     // 是否使用纯hbm模式
};

template <class K, class S>
struct ExportIfPredFunctor {
  __forceinline__ __simt_callee__ bool operator()(const K& key, const S& score,
                                             const K& pattern,
                                             const S& threshold) {
    return score >= threshold;
  }
};

template <class K, class V, class S>
struct ExportIfPredFunctorV2 {
  K pattern;
  S threshold;
  ExportIfPredFunctorV2(K pattern, S threshold)
      : pattern(pattern), threshold(threshold) {}
  template <int GroupSize>
  __forceinline__ __simt_callee__ bool operator()(
      const K& key, const __gm__ V* value, const S& score) {
    /* evaluate key, score and value. */
    return ((!IS_RESERVED_KEY<K>(key)) && (score < threshold));
  }
};

// 通用测试模板函数 - DIM作为模板参数
// 使用模板参数DIM可以正确调用create_continuous_keys函数
// 这样可以避免使用变量传递DIM的问题

template <typename K, typename V, typename S, size_t DIM>
void run_export_batch_if_v2_test(const ExportTestConfig& config) {
  // 1. 初始化环境
  init_env();

  // 2. 检查HBM内存
  size_t total_mem = 0;
  size_t free_mem = 0;
  constexpr size_t hbm_for_values = 1UL << 30;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem
      << "need:" << hbm_for_values;

  // 3. 配置哈希表
  constexpr size_t init_capacity = 128UL * 1024;
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = DIM,
      .io_by_cpu = false,
  };
  if (!config.use_fast_mode) {
    options.max_hbm_for_vectors = init_capacity * sizeof(V) / 2;
  }

  // 4. 初始化哈希表并插入数据
  HashTable<K, V> table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  DeviceData<K, V, S> device_data;
  size_t keynum = config.key_num > 0 ? config.key_num : 1024;
  device_data.malloc(keynum, DIM);

  if (!config.test_empty_table) {
    // 生成测试数据
    vector<K> host_keys(config.key_num);
    vector<V> host_values(config.key_num * DIM);
    create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                         host_values.data(), config.key_num);

    // 分配设备内存并复制数据
    device_data.copy_keys(host_keys, config.key_num);
    device_data.copy_values(host_values, config.key_num, DIM);

    // 插入数据
    table.insert_or_assign(config.key_num, device_data.device_keys, device_data.device_values, nullptr,
                           device_data.stream);
    ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  }

  // 5. 准备导出缓冲区
  const size_t scan_len = table.capacity();
  K* device_export_keys = nullptr;
  V* device_export_values = nullptr;
  S* device_export_scores = nullptr;

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

  // 6. 执行导出操作
  size_t export_count = 0;
  size_t* device_export_count = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_export_count),
                        sizeof(size_t), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMemset(device_export_count, sizeof(size_t), 0, sizeof(size_t)),
      ACL_ERROR_NONE);
  
  K pattern = 0;
  S threshold = 40;
  if (config.functor_version == FunctorVersion::V2) {
    ExportIfPredFunctorV2<K, V, S> pred(pattern, threshold);
    table.template export_batch_if_v2<ExportIfPredFunctorV2<K, V, S>>(
        pred, scan_len, config.offset, device_export_count,
        device_export_keys, device_export_values,
        config.use_scores ? device_export_scores : nullptr, device_data.stream);
  } else if (config.functor_version == FunctorVersion::V1) {
    table.template export_batch_if<ExportIfPredFunctor>(
        pattern, threshold, scan_len, config.offset, device_export_count,
        device_export_keys, device_export_values,
        config.use_scores ? device_export_scores : nullptr, device_data.stream);
    }
  
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(&export_count, sizeof(size_t), device_export_count,
                        sizeof(size_t), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  ASSERT_EQ(aclrtFree(device_export_count), ACL_ERROR_NONE);

 if (config.test_empty_table) {
    ASSERT_EQ(export_count, table.size(device_data.stream));
  } else {
    ASSERT_LE(export_count, table.size(device_data.stream));
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
      V expected_value = static_cast<V>(key) * expected_ratio;

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

  // 8. 清理资源
  ASSERT_EQ(aclrtFree(device_export_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_export_values), ACL_ERROR_NONE);
  if (config.use_scores) {
    ASSERT_EQ(aclrtFree(device_export_scores), ACL_ERROR_NONE);
  }
}

// 辅助宏：简化测试调用
#define RUN_EXPORT_BATCH_IF_V2_TEST(K, V, S, DIM, config) \
  run_export_batch_if_v2_test<K, V, S, DIM>(config)

// 测试用例定义
TEST(test_export_batch_if_v2, test_export_batch_if_v2_basic) {
  ExportTestConfig config = {1024, false, 0, false, FunctorVersion::V2, true};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch_if_v2, test_export_batch_if_v2_empty) {
  ExportTestConfig config = {0, true, 0, false, FunctorVersion::V2, true};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
}

TEST(test_export_batch_if_v2, test_export_batch_if_v2_offset) {
  // 测试不同偏移量下的导出功能
  // 注意：当offset > 0时，不能保证导出数量等于key_num
  ExportTestConfig config = {1024, false, 1000, false, FunctorVersion::V2, true};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  config.offset = 5000;
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch_if_v2, test_export_batch_if_v2_hybrid_basic) {
  ExportTestConfig config = {1024, false, 0, false, FunctorVersion::V2, false};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch_if_v2, test_export_batch_if_v2_hybrid_full_key_num) {
  ExportTestConfig config = {128 * 1024, false, 0, false, FunctorVersion::V2, false};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch_if, test_export_batch_if_basic) {
  ExportTestConfig config = {1024, false, 0, false, FunctorVersion::V1, true};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch_if, test_export_batch_if_full_key_num) {
  ExportTestConfig config = {128 * 1024, false, 0, false, FunctorVersion::V1, true};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch_if, test_export_batch_if_hybrid_basic) {
  ExportTestConfig config = {1024, false, 0, false, FunctorVersion::V1, false};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch_if, test_export_batch_if_hybrid_full_key_num) {
  ExportTestConfig config = {128 * 1024, false, 0, false, FunctorVersion::V1, false};
  RUN_EXPORT_BATCH_IF_V2_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_BATCH_IF_V2_TEST(int64_t, double, uint64_t, 16, config);
}
