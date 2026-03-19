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
#include <vector>
#include "hkv_hashtable.h"
#include "test_device_data.h"
#include "test_template_func.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

TEST(TestForEach, test_copy_specific_key_value) {
  // 1. 初始化
  init_env();
  using K = int64_t;
  using V = float;
  using S = uint64_t;
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t capacity = 1024UL;
  constexpr size_t key_num = 128UL;
  constexpr K target_key = 33;

  // 2. 建表
  auto table = get_table<K, V, S>(dim, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 创建数据并插入
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num);
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_values(host_values, key_num);
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, nullptr,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 5. 创建目标value内存（用于存储找到的value）
  V* target_value = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&target_value),
                        dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(target_value, dim * sizeof(V), 0, dim * sizeof(V)),
            ACL_ERROR_NONE);

  // 6. 定义for_each的执行函数
  ForEachExecutionFunc<K, V, S> func;
  func.target_value = target_value;
  func.dim = dim;
  func.target_key = target_key;

  // 7. 使用for_each遍历并复制目标value
  table->for_each(0, capacity, func, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 8. 验证结果
  vector<V> host_target_value(dim, 0);
  ASSERT_EQ(aclrtMemcpy(host_target_value.data(), dim * sizeof(V), target_value,
                        dim * sizeof(V), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 找到目标key对应的原始value
  vector<V> expected_value(dim, 0);
  for (size_t i = 0; i < key_num; ++i) {
    if (host_keys[i] == target_key) {
      expected_value.assign(host_values.begin() + i * dim,
                            host_values.begin() + (i + 1) * dim);
      break;
    }
  }

  // 比较结果
  ASSERT_EQ(host_target_value, expected_value);

  // 9. 释放资源
  ASSERT_EQ(aclrtFree(target_value), ACL_ERROR_NONE);
}

// 定义参数化测试类
class TestForEach : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override { init_env(); }
};

// 使用INSTANTIATE_TEST_SUITE_P宏定义测试参数
INSTANTIATE_TEST_SUITE_P(TestDimValues, TestForEach,
                         ::testing::Values(1, 8, 16, 32));

TEST_P(TestForEach, test_scores_filter) {
  // 获取当前测试的dim参数
  size_t dim = GetParam();

  // 测试参数
  using K = int64_t;
  using V = float;
  using S = uint64_t;
  constexpr size_t capacity = 1024UL;
  constexpr size_t key_num = 128UL;
  constexpr S threshold = 20;
  size_t search_num = key_num + dim;

  // 2. 创建使用LFU淘汰策略的表
  auto table = get_table<K, V, S, EvictStrategy::kCustomized>(dim, capacity, 1);
  EXPECT_EQ(table->size(), 0);

  // 3. 申请hbm内存
  DeviceData<K, V, S> device_data;
  device_data.malloc(key_num);

  // 4. 创建数据并插入，生成不同的分数
  vector<K> host_keys(key_num, 0);
  vector<S> host_scores(key_num, 0);

  // 生成连续的key
  create_continuous_keys<K, S, V>(host_keys.data(), host_scores.data(), nullptr,
                                  key_num);

  // 生成不同的分数，其中部分大于threshold
  size_t expected_count = 0;
  for (size_t i = 0; i < key_num; ++i) {
    auto hashed_key = test_util::Murmur3HashHost(host_keys[i]);
    auto index = hashed_key % capacity;
    if (index < search_num && host_scores[i] >= threshold) {
      expected_count++;
    }
  }

  // 将数据复制到设备
  device_data.copy_keys(host_keys, key_num);
  device_data.copy_scores(host_scores, key_num);

  // 插入数据
  table->insert_or_assign(key_num, device_data.device_keys,
                          device_data.device_values, device_data.device_scores,
                          device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);
  EXPECT_EQ(table->size(device_data.stream), key_num);

  // 5. 创建计数器内存
  uint64_t* device_count = nullptr;
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_count),
                        sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(device_count, sizeof(uint64_t), 0, sizeof(uint64_t)),
            ACL_ERROR_NONE);

  // 6. 定义for_each的执行函数
  ForEachScoresFilterFunc<K, V, S> func;
  func.count = device_count;
  func.threshold = threshold;  // 转换为与存储格式相同的阈值

  // 7. 使用for_each遍历并统计符合条件的数量
  table->for_each(0, search_num, func, device_data.stream);
  ASSERT_EQ(aclrtSynchronizeStream(device_data.stream), ACL_ERROR_NONE);

  // 8. 验证结果
  uint64_t host_count = 0;
  ASSERT_EQ(aclrtMemcpy(&host_count, sizeof(uint64_t), device_count,
                        sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 比较结果
  ASSERT_EQ(host_count, expected_count);

  // 9. 释放资源
  ASSERT_EQ(aclrtFree(device_count), ACL_ERROR_NONE);
}
