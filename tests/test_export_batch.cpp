/* *
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

// 通用测试模板函数 - DIM作为模板参数
// 使用模板参数DIM可以正确调用create_continuous_keys函数
// 这样可以避免使用变量传递DIM的问题

template <typename K, typename V, typename S, size_t DIM>
void run_export_test(const ExportTestConfig& config) {
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

  // 4. 初始化哈希表并插入数据
  HashTable<K, V> table;
  table.init(options);

  K* device_keys = nullptr;
  V* device_values = nullptr;

  if (!config.test_empty_table) {
    // 生成测试数据
    vector<K> host_keys(config.key_num);
    vector<V> host_values(config.key_num * DIM);
    create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                         host_values.data(), config.key_num);

    // 分配设备内存并复制数据
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

    // 插入数据
    aclrtStream stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
    table.insert_or_assign(config.key_num, device_keys, device_values, nullptr,
                           stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
  }

  // 5. 准备导出缓冲区
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

  // 6. 执行导出操作
  size_t export_count = 0;

  if (config.test_overload) {
    // 测试重载版本
    export_count = table.export_batch(
        scan_len, config.offset, device_export_keys, device_export_values,
        config.use_scores ? device_export_scores : nullptr, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  } else {
    // 测试普通版本
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

  // 7. 验证结果
  if (config.test_empty_table) {
    // 空表测试：导出数量应为0
    ASSERT_EQ(export_count, 0);
  } else if (config.offset == 0) {
    // 无偏移测试：导出数量应等于插入数量
    ASSERT_EQ(export_count, config.key_num);
  } else {
    // 偏移测试：导出数量可能小于插入数量，但不能大于插入数量
    ASSERT_LE(export_count, config.key_num);
    ASSERT_GE(export_count, 0);
  }

  if (export_count > 0 && !config.test_empty_table) {
    // 复制导出结果到主机
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

    // 构建验证map
    vector<K> host_keys(config.key_num);
    vector<V> host_values(config.key_num * DIM);
    create_continuous_keys<K, S, V, DIM>(host_keys.data(), nullptr,
                                         host_values.data(), config.key_num);

    unordered_map<K, vector<V>> reference_map;
    reference_map.reserve(config.key_num);
    for (size_t i = 0; i < config.key_num; ++i) {
      vector<V> vec(DIM);
      copy(host_values.begin() + i * DIM, host_values.begin() + (i + 1) * DIM,
           vec.begin());
      reference_map[host_keys[i]] = move(vec);
    }

    // 验证每个导出的键值对都存在于参考map中且值正确
    for (size_t i = 0; i < export_count; ++i) {
      const K& key = host_export_keys[i];
      auto it = reference_map.find(key);
      ASSERT_NE(it, reference_map.end())
          << "Exported key " << key << " not found in reference map";

      vector<V> exported_vec(DIM);
      copy(host_export_values.begin() + i * DIM,
           host_export_values.begin() + (i + 1) * DIM, exported_vec.begin());
      ASSERT_EQ(exported_vec, it->second) << "Values mismatch for key " << key;
    }
  }

  // 8. 清理资源
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

// 辅助宏：简化测试调用
#define RUN_EXPORT_TEST(K, V, S, DIM, config) \
  run_export_test<K, V, S, DIM>(config)

// 测试用例定义
TEST(test_export_batch, test_export_basic) {
  ExportTestConfig config = {1024, false, false, 0, false};
  RUN_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
  RUN_EXPORT_TEST(int64_t, double, uint64_t, 16, config);
}

TEST(test_export_batch, test_export_overload) {
  ExportTestConfig config = {1024, true, false, 0, true};
  RUN_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
}

TEST(test_export_batch, test_export_empty) {
  ExportTestConfig config = {0, false, true, 0, false};
  RUN_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
}

TEST(test_export_batch, test_export_offset) {
  // 测试不同偏移量下的导出功能
  // 注意：当offset > 0时，不能保证导出数量等于key_num
  ExportTestConfig config = {1024, false, false, 1000, false};
  RUN_EXPORT_TEST(uint64_t, float, uint64_t, 8, config);
  config.offset = 5000;
  RUN_EXPORT_TEST(int64_t, double, uint64_t, 16, config);
}
