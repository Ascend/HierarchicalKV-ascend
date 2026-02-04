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
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
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

// 使用Customized淘汰策略，手动控制分数
constexpr int EVICT_STRATEGY = EvictStrategy::kCustomized;

class InsertAndEvictTest : public ::testing::Test {
 public:
  static constexpr size_t DEFAULT_DIM = 8;
  static constexpr size_t DEFAULT_INIT_CAPACITY = 128UL * 1024;
  static constexpr size_t DEFAULT_MAX_BUCKET_SIZE = 128;
  static constexpr size_t DEFAULT_HBM_FOR_VALUES = 1UL << 30;

  void SetUp() override {
    init_env();

    size_t total_mem = 0;
    size_t free_mem = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
              ACL_ERROR_NONE);
    ASSERT_GT(free_mem, DEFAULT_HBM_FOR_VALUES);
  }

  void TearDown() override {}

  // ======== 辅助函数：创建和初始化哈希表 ========
  template <typename TableType>
  void InitTable(TableType& table, size_t dim, size_t init_capacity,
                 size_t hbm_for_values, size_t max_bucket_size = DEFAULT_MAX_BUCKET_SIZE) {
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .max_bucket_size = max_bucket_size,
        .dim = dim,
        .io_by_cpu = false,
    };
    table.init(options);
  }

  // ======== 辅助函数：设备内存管理 ========
  struct DeviceMemory {
    K* keys = nullptr;
    V* values = nullptr;
    S* scores = nullptr;
    K* evicted_keys = nullptr;
    V* evicted_values = nullptr;
    S* evicted_scores = nullptr;
    uint64_t* evicted_counter = nullptr;
    V** values_ptr = nullptr;
    bool* found = nullptr;
    aclrtStream stream = nullptr;

    void AllocateForInsert(size_t key_num, size_t dim) {
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&keys),
                            key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&values),
                            key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&scores),
                            key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&evicted_keys),
                            key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&evicted_values),
                            key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&evicted_scores),
                            key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&evicted_counter),
                            sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);

      // 初始化 evicted_counter 为 0
      ASSERT_EQ(aclrtMemset(evicted_counter, sizeof(uint64_t), 0, sizeof(uint64_t)),
                ACL_ERROR_NONE);

      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
    }

    void AllocateForFind(size_t key_num, size_t dim) {
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&keys),
                            key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&values_ptr),
                            key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&found),
                            key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&scores),
                            key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);
      ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&values),
                            key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
                ACL_ERROR_NONE);

      ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
    }

    ~DeviceMemory() {
      if (keys) aclrtFree(keys);
      if (values) aclrtFree(values);
      if (scores) aclrtFree(scores);
      if (evicted_keys) aclrtFree(evicted_keys);
      if (evicted_values) aclrtFree(evicted_values);
      if (evicted_scores) aclrtFree(evicted_scores);
      if (evicted_counter) aclrtFree(evicted_counter);
      if (values_ptr) aclrtFree(values_ptr);
      if (found) aclrtFree(found);
      if (stream) {
        aclrtSynchronizeStream(stream);
        aclrtDestroyStream(stream);
      }
    }
  };

  // ======== 辅助函数：拷贝数据到设备 ========
  void CopyToDevice(const vector<K>& host_keys,
                    const vector<V>& host_values,
                    const vector<S>& host_scores,
                    DeviceMemory& device_mem,
                    size_t key_num, size_t dim) {
    ASSERT_EQ(aclrtMemcpy(device_mem.keys, key_num * sizeof(K),
                          host_keys.data(), key_num * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(device_mem.values, key_num * dim * sizeof(V),
                          host_values.data(), key_num * dim * sizeof(V),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
    if (!host_scores.empty()) {
      ASSERT_EQ(aclrtMemcpy(device_mem.scores, key_num * sizeof(S),
                            host_scores.data(), key_num * sizeof(S),
                            ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
    }
  }

  // ======== 辅助函数：验证插入结果（通过find） ========
  template <typename TableType>
  void VerifyInsertByFind(TableType& table,
                          const vector<K>& host_keys,
                          const vector<V>& expected_values,
                          const vector<S>& expected_scores,
                          size_t key_num, size_t dim,
                          size_t expected_found_num = SIZE_MAX) {
    if (expected_found_num == SIZE_MAX) {
      expected_found_num = key_num;
    }

    DeviceMemory find_mem;
    find_mem.AllocateForFind(key_num, dim);

    // 拷贝keys到设备
    ASSERT_EQ(aclrtMemcpy(find_mem.keys, key_num * sizeof(K),
                          host_keys.data(), key_num * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);

    // 调用find
    table.find(key_num, find_mem.keys, find_mem.values_ptr, find_mem.found,
               find_mem.scores, find_mem.stream, true);
    ASSERT_EQ(aclrtSynchronizeStream(find_mem.stream), ACL_ERROR_NONE);

    // 读取结果
    vector<uint8_t> host_found(key_num);
    vector<void*> host_values_ptr(key_num);
    vector<S> host_scores(key_num);

    ASSERT_EQ(aclrtMemcpy(host_found.data(), key_num * sizeof(bool),
                          find_mem.found, key_num * sizeof(bool),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(host_values_ptr.data(), key_num * sizeof(V*),
                          find_mem.values_ptr, key_num * sizeof(V*),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMemcpy(host_scores.data(), key_num * sizeof(S),
                          find_mem.scores, key_num * sizeof(S),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    // 验证找到的数量
    size_t found_num = 0;
    for (size_t i = 0; i < key_num; i++) {
      if (host_found[i]) {
        found_num++;
        ASSERT_NE(host_values_ptr[i], nullptr);

        // 验证values
        vector<V> real_values(dim);
        ASSERT_EQ(aclrtMemcpy(real_values.data(), dim * sizeof(V),
                              host_values_ptr[i], dim * sizeof(V),
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  ACL_ERROR_NONE);

        vector<V> expect_values(expected_values.begin() + i * dim,
                                expected_values.begin() + (i + 1) * dim);
        EXPECT_EQ(expect_values, real_values)
            << "Key " << host_keys[i] << " values mismatch";

        // 验证scores
        if (!expected_scores.empty()) {
          EXPECT_EQ(expected_scores[i], host_scores[i])
              << "Key " << host_keys[i] << " score mismatch";
        }
      }
    }
    EXPECT_EQ(found_num, expected_found_num);
  }

  // ======== 辅助函数：验证淘汰结果 ========
  struct EvictResult {
    vector<K> keys;
    vector<V> values;
    vector<S> scores;
    uint64_t counter;
  };

  EvictResult GetEvictResult(DeviceMemory& device_mem,
                              size_t max_key_num, size_t dim) {
    EvictResult result;

    // 读取淘汰计数器
    EXPECT_EQ(aclrtMemcpy(&result.counter, sizeof(uint64_t),
                          device_mem.evicted_counter, sizeof(uint64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    if (result.counter == 0) {
      return result;
    }

    EXPECT_LE(result.counter, max_key_num)
        << "Evicted counter exceeds max";

    // 读取淘汰的keys
    result.keys.resize(result.counter);
    EXPECT_EQ(aclrtMemcpy(result.keys.data(), result.counter * sizeof(K),
                          device_mem.evicted_keys, result.counter * sizeof(K),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    // 读取淘汰的values
    result.values.resize(result.counter * dim);
    EXPECT_EQ(aclrtMemcpy(result.values.data(),
                          result.counter * dim * sizeof(V),
                          device_mem.evicted_values,
                          result.counter * dim * sizeof(V),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    // 读取淘汰的scores
    result.scores.resize(result.counter);
    EXPECT_EQ(aclrtMemcpy(result.scores.data(), result.counter * sizeof(S),
                          device_mem.evicted_scores, result.counter * sizeof(S),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);

    return result;
  }
};

// 测试用例1: 插入场景 - 正常插入新key
TEST_F(InsertAndEvictTest, BasicInsert_SmallScale) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;
  constexpr size_t init_capacity = DEFAULT_INIT_CAPACITY;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  // 准备测试数据
  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  vector<S> host_scores(key_num);
  create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                              host_values.data(), key_num);

  // 分配设备内存
  DeviceMemory device_mem;
  device_mem.AllocateForInsert(key_num, dim);

  // 拷贝数据到设备
  CopyToDevice(host_keys, host_values, host_scores, device_mem, key_num, dim);

  // 调用 insert_and_evict
  table.insert_and_evict(key_num, device_mem.keys, device_mem.values,
                         device_mem.scores, device_mem.evicted_keys,
                         device_mem.evicted_values, device_mem.evicted_scores,
                         device_mem.evicted_counter, device_mem.stream,
                         true, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_mem.stream), ACL_ERROR_NONE);

  // 验证：没有淘汰发生
  auto evict_result = GetEvictResult(device_mem, key_num, dim);
  EXPECT_EQ(evict_result.counter, 0) << "No eviction should happen";

  // 验证：通过find检查所有keys都正确插入
  VerifyInsertByFind(table, host_keys, host_values, host_scores,
                     key_num, dim, key_num);
}

// 测试用例2: Assign场景 - 更新已存在的key
TEST_F(InsertAndEvictTest, UpdateExisting_VerifyNewValues) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 512;
  constexpr size_t init_capacity = DEFAULT_INIT_CAPACITY;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  // 第一次插入：原始数据
  vector<K> host_keys(key_num);
  vector<V> host_values_old(key_num * dim);
  vector<S> host_scores_old(key_num);
  create_random_keys<K, S, V>(dim, host_keys.data(), host_scores_old.data(),
                              host_values_old.data(), key_num);

  DeviceMemory device_mem1;
  device_mem1.AllocateForInsert(key_num, dim);
  CopyToDevice(host_keys, host_values_old, host_scores_old,
               device_mem1, key_num, dim);

  table.insert_and_evict(key_num, device_mem1.keys, device_mem1.values,
                         device_mem1.scores, device_mem1.evicted_keys,
                         device_mem1.evicted_values, device_mem1.evicted_scores,
                         device_mem1.evicted_counter, device_mem1.stream,
                         true, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_mem1.stream), ACL_ERROR_NONE);

  // 验证第一次插入成功
  auto evict_result1 = GetEvictResult(device_mem1, key_num, dim);
  EXPECT_EQ(evict_result1.counter, 0);

  // 第二次插入：相同keys，新的values和scores
  vector<V> host_values_new(key_num * dim);
  vector<S> host_scores_new(key_num);
  for (size_t i = 0; i < key_num; i++) {
    host_scores_new[i] = host_scores_old[i] + 1000;  // 新分数
    for (size_t j = 0; j < dim; j++) {
      host_values_new[i * dim + j] = host_values_old[i * dim + j] + 100.0f;  // 新值
    }
  }

  DeviceMemory device_mem2;
  device_mem2.AllocateForInsert(key_num, dim);
  CopyToDevice(host_keys, host_values_new, host_scores_new,
               device_mem2, key_num, dim);

  table.insert_and_evict(key_num, device_mem2.keys, device_mem2.values,
                         device_mem2.scores, device_mem2.evicted_keys,
                         device_mem2.evicted_values, device_mem2.evicted_scores,
                         device_mem2.evicted_counter, device_mem2.stream,
                         true, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_mem2.stream), ACL_ERROR_NONE);

  // 验证：没有淘汰（因为是更新已存在的key）
  auto evict_result2 = GetEvictResult(device_mem2, key_num, dim);
  EXPECT_EQ(evict_result2.counter, 0) << "Update should not evict";

  // 验证：通过find检查所有keys的values和scores都更新为新值
  VerifyInsertByFind(table, host_keys, host_values_new, host_scores_new,
                     key_num, dim, key_num);
}

// 测试用例3: 淘汰场景 - 满桶全部淘汰
TEST_F(InsertAndEvictTest, EvictAllLowScoreKeys) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t bucket_size = 128;
  constexpr size_t bucket_idx = 0;
  constexpr size_t init_capacity = bucket_size * 4;  // 4个桶
  constexpr size_t freq_range = 1000;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES, bucket_size);

  // 第一步：填满一个桶（bucket_idx=0），使用低分数 [0, 127]
  vector<K> host_keys_low(bucket_size);
  vector<V> host_values_low(bucket_size * dim);
  vector<S> host_scores_low(bucket_size);
  K first_batch_min = 1;
  K first_batch_max = 10000000;
  
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      host_keys_low.data(), host_scores_low.data(), host_values_low.data(),
      bucket_size, init_capacity, bucket_size, bucket_idx,
      first_batch_min, first_batch_max, freq_range);

  // 手动设置第一批为低分数 [0, 127]
  for (size_t i = 0; i < bucket_size; i++) {
    host_scores_low[i] = i;  // 分数 [0, 1, 2, ..., 127]
  }

  DeviceMemory device_mem1;
  device_mem1.AllocateForInsert(bucket_size, dim);
  CopyToDevice(host_keys_low, host_values_low, host_scores_low,
               device_mem1, bucket_size, dim);

  table.insert_and_evict(bucket_size, device_mem1.keys, device_mem1.values,
                         device_mem1.scores, device_mem1.evicted_keys,
                         device_mem1.evicted_values, device_mem1.evicted_scores,
                         device_mem1.evicted_counter, device_mem1.stream,
                         true, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_mem1.stream), ACL_ERROR_NONE);

  // 验证第一次插入无淘汰
  auto evict_result1 = GetEvictResult(device_mem1, bucket_size, dim);
  EXPECT_EQ(evict_result1.counter, 0) << "First insert should not evict";

  // 第二步：插入128个新的高分keys，替换掉全部低分keys
  size_t new_key_num = bucket_size;  // 与第一批相同数量
  vector<K> host_keys_high(new_key_num);
  vector<V> host_values_high(new_key_num * dim);
  vector<S> host_scores_high(new_key_num);

  // 生成新keys（key范围完全不重叠）
  K second_batch_min = 100000000;
  K second_batch_max = 200000000;
  
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      host_keys_high.data(), host_scores_high.data(), host_values_high.data(),
      new_key_num, init_capacity, bucket_size, bucket_idx,
      second_batch_min, second_batch_max, freq_range);

  // 手动设置高分数 [500, 627]，确保高于第一批
  for (size_t i = 0; i < new_key_num; i++) {
    host_scores_high[i] = 500 + i;  // [500, 501, ..., 627]
  }

  DeviceMemory device_mem2;
  device_mem2.AllocateForInsert(new_key_num, dim);
  CopyToDevice(host_keys_high, host_values_high, host_scores_high,
               device_mem2, new_key_num, dim);

  table.insert_and_evict(new_key_num, device_mem2.keys, device_mem2.values,
                         device_mem2.scores, device_mem2.evicted_keys,
                         device_mem2.evicted_values, device_mem2.evicted_scores,
                         device_mem2.evicted_counter, device_mem2.stream,
                         true, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_mem2.stream), ACL_ERROR_NONE);

  // 验证：所有第一批的keys都被淘汰
  auto evict_result2 = GetEvictResult(device_mem2, new_key_num, dim);
  EXPECT_EQ(evict_result2.counter, bucket_size) 
      << "Should evict all " << bucket_size << " old keys";

  // 验证：所有被淘汰的keys都来自第一批
  unordered_set<K> first_batch_keys(host_keys_low.begin(), host_keys_low.end());
  for (size_t i = 0; i < evict_result2.counter; i++) {
    K evicted_key = evict_result2.keys[i];
    EXPECT_TRUE(first_batch_keys.find(evicted_key) != first_batch_keys.end())
        << "Evicted key " << evicted_key << " should be from first batch";
  }

  // 验证：所有被淘汰的分数都是低分 [0, 127]
  for (size_t i = 0; i < evict_result2.counter; i++) {
    EXPECT_LT(evict_result2.scores[i], 128)
        << "Evicted score " << evict_result2.scores[i] << " should be low";
  }

  // 验证：被淘汰的keys对应的values正确
  unordered_map<K, vector<V>> original_values_map;
  for (size_t i = 0; i < bucket_size; i++) {
    vector<V> values_vec(host_values_low.begin() + i * dim,
                         host_values_low.begin() + (i + 1) * dim);
    original_values_map[host_keys_low[i]] = values_vec;
  }

  for (size_t i = 0; i < evict_result2.counter; i++) {
    K evicted_key = evict_result2.keys[i];
    ASSERT_TRUE(original_values_map.find(evicted_key) != original_values_map.end())
        << "Evicted key " << evicted_key << " not found in original map";
    
    vector<V> expected_values = original_values_map[evicted_key];
    vector<V> actual_values(evict_result2.values.begin() + i * dim,
                            evict_result2.values.begin() + (i + 1) * dim);
    EXPECT_EQ(expected_values, actual_values)
        << "Evicted values for key " << evicted_key << " mismatch";
  }

  // 验证：新插入的所有高分keys都在表中
  VerifyInsertByFind(table, host_keys_high, host_values_high, host_scores_high,
                     new_key_num, dim, new_key_num);
}

// 测试用例4: 边界测试 - 空插入
TEST_F(InsertAndEvictTest, BoundaryTest_EmptyInsert) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 0;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  K* device_keys = nullptr;
  V* device_values = nullptr;
  S* device_scores = nullptr;
  K* device_evicted_keys = nullptr;
  V* device_evicted_values = nullptr;
  S* device_evicted_scores = nullptr;
  uint64_t* device_evicted_counter = nullptr;

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 调用 insert_and_evict，n=0 应该直接返回
  table.insert_and_evict(key_num, device_keys, device_values, device_scores,
                         device_evicted_keys, device_evicted_values,
                         device_evicted_scores, device_evicted_counter,
                         stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);

  // 验证：函数正常返回，无崩溃
  SUCCEED();
}

// 测试用例5: 大规模插入测试
TEST_F(InsertAndEvictTest, LargeScale_Insert) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 64 * 1024;
  constexpr size_t init_capacity = DEFAULT_INIT_CAPACITY;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num);
  vector<V> host_values(key_num * dim);
  vector<S> host_scores(key_num);
  create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                              host_values.data(), key_num);

  DeviceMemory device_mem;
  device_mem.AllocateForInsert(key_num, dim);
  CopyToDevice(host_keys, host_values, host_scores, device_mem, key_num, dim);

  table.insert_and_evict(key_num, device_mem.keys, device_mem.values,
                         device_mem.scores, device_mem.evicted_keys,
                         device_mem.evicted_values, device_mem.evicted_scores,
                         device_mem.evicted_counter, device_mem.stream,
                         true, false);
  ASSERT_EQ(aclrtSynchronizeStream(device_mem.stream), ACL_ERROR_NONE);

  // 验证：大规模插入成功
  auto evict_result = GetEvictResult(device_mem, key_num, dim);
  EXPECT_GE(evict_result.counter, 0);  // 可能有淘汰也可能没有

  // 验证所有 keys
  VerifyInsertByFind(table, host_keys, host_values, host_scores,
                     key_num, dim, key_num);
}

// 测试用例6: 用 size_type insert_and_evict
// 1) 首次插入：返回淘汰数为0，find验证(K,V,S)正确
// 2) 二次插入相同keys：返回淘汰数为0，find验证(V,S)已更新
TEST_F(InsertAndEvictTest, InsertAndEvict_ReturnSizeType_InsertAndUpdate) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 512;
  constexpr size_t init_capacity = DEFAULT_INIT_CAPACITY;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  // 第一轮：插入新key
  vector<K> host_keys(key_num);
  vector<V> host_values_old(key_num * dim);
  vector<S> host_scores_old(key_num);
  create_random_keys<K, S, V>(dim, host_keys.data(), host_scores_old.data(),
                              host_values_old.data(), key_num);

  DeviceMemory device_mem;
  device_mem.AllocateForInsert(key_num, dim);
  CopyToDevice(host_keys, host_values_old, host_scores_old, device_mem, key_num, dim);

  // size_type 版本：直接返回淘汰数量（内部会做同步）
  size_t evicted_count1 = table.insert_and_evict(
      key_num, device_mem.keys, device_mem.values, device_mem.scores,
      device_mem.evicted_keys, device_mem.evicted_values, device_mem.evicted_scores,
      device_mem.stream, true, false);
  EXPECT_EQ(evicted_count1, 0) << "First insert should not evict";

  // find验证首轮插入(K,V,S)
  VerifyInsertByFind(table, host_keys, host_values_old, host_scores_old,
                     key_num, dim, key_num);

  // 第二轮：相同keys，更新values和scores
  vector<V> host_values_new(key_num * dim);
  vector<S> host_scores_new(key_num);
  for (size_t i = 0; i < key_num; i++) {
    host_scores_new[i] = host_scores_old[i] + 1000;
    for (size_t j = 0; j < dim; j++) {
      host_values_new[i * dim + j] = host_values_old[i * dim + j] + 100.0f;
    }
  }

  // 复用同一块设备输入buffer，写入第二轮数据
  ASSERT_EQ(aclrtMemcpy(device_mem.keys, key_num * sizeof(K),
                        host_keys.data(), key_num * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_mem.values, key_num * dim * sizeof(V),
                        host_values_new.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_mem.scores, key_num * sizeof(S),
                        host_scores_new.data(), key_num * sizeof(S),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  size_t evicted_count2 = table.insert_and_evict(
      key_num, device_mem.keys, device_mem.values, device_mem.scores,
      device_mem.evicted_keys, device_mem.evicted_values, device_mem.evicted_scores,
      device_mem.stream, true, false);
  EXPECT_EQ(evicted_count2, 0) << "Update existing keys should not evict";

  // find验证二轮更新后的(V,S)
  VerifyInsertByFind(table, host_keys, host_values_new, host_scores_new,
                     key_num, dim, key_num);
}

// 测试用例7: 拒绝准入（score < min_score）
// 目标：桶已满且新key分数更低时，新key不准入；evicted_* 输出应记录“被拒绝的新key”
TEST_F(InsertAndEvictTest, RejectAdmission_LowScoreShouldNotEnter) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t bucket_size = 128;
  constexpr size_t bucket_idx = 0;
  constexpr size_t init_capacity = bucket_size * 4;  // 4个桶

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES, bucket_size);

  // 第一步：填满桶（高分），确保 min_score >= 500
  vector<K> host_keys_high(bucket_size);
  vector<V> host_values_high(bucket_size * dim);
  vector<S> host_scores_high(bucket_size);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      host_keys_high.data(), host_scores_high.data(), host_values_high.data(),
      bucket_size, init_capacity, bucket_size, bucket_idx,
      1, 10000000, 1000);
  for (size_t i = 0; i < bucket_size; i++) {
    host_scores_high[i] = 500 + i;  // [500, 627]
  }

  DeviceMemory fill_mem;
  fill_mem.AllocateForInsert(bucket_size, dim);
  CopyToDevice(host_keys_high, host_values_high, host_scores_high,
               fill_mem, bucket_size, dim);

  table.insert_and_evict(bucket_size, fill_mem.keys, fill_mem.values, fill_mem.scores,
                         fill_mem.evicted_keys, fill_mem.evicted_values, fill_mem.evicted_scores,
                         fill_mem.evicted_counter, fill_mem.stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(fill_mem.stream), ACL_ERROR_NONE);
  auto fill_evict = GetEvictResult(fill_mem, bucket_size, dim);
  EXPECT_EQ(fill_evict.counter, 0);

  // 第二步：插入低分新key，触发拒绝准入
  constexpr size_t new_key_num = 64;
  vector<K> host_keys_low(new_key_num);
  vector<V> host_values_low(new_key_num * dim);
  vector<S> host_scores_low(new_key_num);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      host_keys_low.data(), host_scores_low.data(), host_values_low.data(),
      new_key_num, init_capacity, bucket_size, bucket_idx,
      100000000, 200000000, 1000);
  for (size_t i = 0; i < new_key_num; i++) {
    host_scores_low[i] = i;  // [0, 63] < 500，保证 score < min_score
  }

  DeviceMemory reject_mem;
  reject_mem.AllocateForInsert(new_key_num, dim);
  CopyToDevice(host_keys_low, host_values_low, host_scores_low,
               reject_mem, new_key_num, dim);

  table.insert_and_evict(new_key_num, reject_mem.keys, reject_mem.values, reject_mem.scores,
                         reject_mem.evicted_keys, reject_mem.evicted_values, reject_mem.evicted_scores,
                         reject_mem.evicted_counter, reject_mem.stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(reject_mem.stream), ACL_ERROR_NONE);

  auto reject_result = GetEvictResult(reject_mem, new_key_num, dim);
  EXPECT_EQ(reject_result.counter, new_key_num) << "All low-score keys should be rejected";

  // evicted_keys 应全部来自第二批（被拒绝的新key）
  unordered_set<K> low_set(host_keys_low.begin(), host_keys_low.end());
  for (size_t i = 0; i < reject_result.counter; i++) {
    EXPECT_TRUE(low_set.find(reject_result.keys[i]) != low_set.end())
        << "Evicted(rejected) key should be from low-score batch";
  }

  // evicted_values 应等于输入 values
  unordered_map<K, vector<V>> low_values_map;
  for (size_t i = 0; i < new_key_num; i++) {
    low_values_map[host_keys_low[i]] = vector<V>(host_values_low.begin() + i * dim,
                                                 host_values_low.begin() + (i + 1) * dim);
  }
  for (size_t i = 0; i < reject_result.counter; i++) {
    const K k = reject_result.keys[i];
    auto it = low_values_map.find(k);
    ASSERT_TRUE(it != low_values_map.end());
    vector<V> actual(reject_result.values.begin() + i * dim,
                     reject_result.values.begin() + (i + 1) * dim);
    EXPECT_EQ(it->second, actual) << "Rejected key values mismatch";
  }

  // find：第一批仍然存在（桶未变化）
  VerifyInsertByFind(table, host_keys_high, host_values_high, host_scores_high,
                     bucket_size, dim, bucket_size);
  // find：第二批全部不存在
  VerifyInsertByFind(table, host_keys_low, host_values_low, host_scores_low,
                     new_key_num, dim, 0);
}

// 测试用例8: 混合批次（同一批同时包含 update / insert / evict / reject）
TEST_F(InsertAndEvictTest, MixedBatch_InsertUpdateEvictReject) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t bucket_size = 128;
  constexpr size_t init_capacity = bucket_size * 4;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES, bucket_size);

  // 预置：填满 bucket0（用于 evict/reject），分数设置为 [1000, 1127]
  constexpr size_t bucket_idx_full = 0;
  vector<K> bucket0_keys(bucket_size);
  vector<V> bucket0_values(bucket_size * dim);
  vector<S> bucket0_scores(bucket_size);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      bucket0_keys.data(), bucket0_scores.data(), bucket0_values.data(),
      bucket_size, init_capacity, bucket_size, bucket_idx_full,
      1, 10000000, 1000);
  for (size_t i = 0; i < bucket_size; i++) {
    bucket0_scores[i] = 1000 + i;
  }

  DeviceMemory prefill0_mem;
  prefill0_mem.AllocateForInsert(bucket_size, dim);
  CopyToDevice(bucket0_keys, bucket0_values, bucket0_scores, prefill0_mem, bucket_size, dim);
  table.insert_and_evict(bucket_size, prefill0_mem.keys, prefill0_mem.values, prefill0_mem.scores,
                         prefill0_mem.evicted_keys, prefill0_mem.evicted_values, prefill0_mem.evicted_scores,
                         prefill0_mem.evicted_counter, prefill0_mem.stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(prefill0_mem.stream), ACL_ERROR_NONE);
  EXPECT_EQ(GetEvictResult(prefill0_mem, bucket_size, dim).counter, 0);

  // 预置：插入一批 key_A 到 bucket1（用于 update）
  constexpr size_t bucket_idx_update = 1;
  constexpr size_t update_num = 32;
  vector<K> keys_A(update_num);
  vector<V> values_A_old(update_num * dim);
  vector<S> scores_A_old(update_num);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      keys_A.data(), scores_A_old.data(), values_A_old.data(),
      update_num, init_capacity, bucket_size, bucket_idx_update,
      200000000, 210000000, 1000);
  for (size_t i = 0; i < update_num; i++) {
    scores_A_old[i] = 10 + i;
  }

  DeviceMemory prefillA_mem;
  prefillA_mem.AllocateForInsert(update_num, dim);
  CopyToDevice(keys_A, values_A_old, scores_A_old, prefillA_mem, update_num, dim);
  table.insert_and_evict(update_num, prefillA_mem.keys, prefillA_mem.values, prefillA_mem.scores,
                         prefillA_mem.evicted_keys, prefillA_mem.evicted_values, prefillA_mem.evicted_scores,
                         prefillA_mem.evicted_counter, prefillA_mem.stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(prefillA_mem.stream), ACL_ERROR_NONE);
  EXPECT_EQ(GetEvictResult(prefillA_mem, update_num, dim).counter, 0);

  // 构造混合 batch：A(update) + B(insert 到 bucket2) + C(evict bucket0) + D(reject bucket0)
  constexpr size_t insert_num = 32;
  constexpr size_t evict_num = 16;
  constexpr size_t reject_num = 16;
  constexpr size_t bucket_idx_insert = 2;

  // A(update)：同 keys_A，写入新 values/scores
  vector<V> values_A_new(update_num * dim);
  vector<S> scores_A_new(update_num);
  for (size_t i = 0; i < update_num; i++) {
    scores_A_new[i] = scores_A_old[i] + 1000;
    for (size_t j = 0; j < dim; j++) {
      values_A_new[i * dim + j] = values_A_old[i * dim + j] + 100.0f;
    }
  }

  // B(insert)：落 bucket2，保证空位插入
  vector<K> keys_B(insert_num);
  vector<V> values_B(insert_num * dim);
  vector<S> scores_B(insert_num);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      keys_B.data(), scores_B.data(), values_B.data(),
      insert_num, init_capacity, bucket_size, bucket_idx_insert,
      300000000, 310000000, 1000);
  for (size_t i = 0; i < insert_num; i++) {
    scores_B[i] = 200 + i;
  }

  // C(evict)：落 bucket0 且高分，确保准入并淘汰旧 key
  vector<K> keys_C(evict_num);
  vector<V> values_C(evict_num * dim);
  vector<S> scores_C(evict_num);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      keys_C.data(), scores_C.data(), values_C.data(),
      evict_num, init_capacity, bucket_size, bucket_idx_full,
      100000000, 110000000, 1000);
  for (size_t i = 0; i < evict_num; i++) {
    scores_C[i] = 2000 + i;  // > 1000，保证不拒绝
  }

  // D(reject)：落 bucket0 且低分，确保拒绝准入
  vector<K> keys_D(reject_num);
  vector<V> values_D(reject_num * dim);
  vector<S> scores_D(reject_num);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      keys_D.data(), scores_D.data(), values_D.data(),
      reject_num, init_capacity, bucket_size, bucket_idx_full,
      400000000, 410000000, 1000);
  for (size_t i = 0; i < reject_num; i++) {
    scores_D[i] = i;  // < 1000，保证拒绝
  }

  // 拼接 batch（顺序不重要，断言不依赖输出顺序）
  const size_t n = update_num + insert_num + evict_num + reject_num;
  vector<K> batch_keys;
  vector<V> batch_values;
  vector<S> batch_scores;
  batch_keys.reserve(n);
  batch_values.reserve(n * dim);
  batch_scores.reserve(n);

  batch_keys.insert(batch_keys.end(), keys_A.begin(), keys_A.end());
  batch_values.insert(batch_values.end(), values_A_new.begin(), values_A_new.end());
  batch_scores.insert(batch_scores.end(), scores_A_new.begin(), scores_A_new.end());

  batch_keys.insert(batch_keys.end(), keys_B.begin(), keys_B.end());
  batch_values.insert(batch_values.end(), values_B.begin(), values_B.end());
  batch_scores.insert(batch_scores.end(), scores_B.begin(), scores_B.end());

  batch_keys.insert(batch_keys.end(), keys_C.begin(), keys_C.end());
  batch_values.insert(batch_values.end(), values_C.begin(), values_C.end());
  batch_scores.insert(batch_scores.end(), scores_C.begin(), scores_C.end());

  batch_keys.insert(batch_keys.end(), keys_D.begin(), keys_D.end());
  batch_values.insert(batch_values.end(), values_D.begin(), values_D.end());
  batch_scores.insert(batch_scores.end(), scores_D.begin(), scores_D.end());

  DeviceMemory mixed_mem;
  mixed_mem.AllocateForInsert(n, dim);
  CopyToDevice(batch_keys, batch_values, batch_scores, mixed_mem, n, dim);

  table.insert_and_evict(n, mixed_mem.keys, mixed_mem.values, mixed_mem.scores,
                         mixed_mem.evicted_keys, mixed_mem.evicted_values, mixed_mem.evicted_scores,
                         mixed_mem.evicted_counter, mixed_mem.stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(mixed_mem.stream), ACL_ERROR_NONE);

  auto mixed_evict = GetEvictResult(mixed_mem, n, dim);
  EXPECT_EQ(mixed_evict.counter, evict_num + reject_num);

  unordered_set<K> bucket0_set(bucket0_keys.begin(), bucket0_keys.end());
  unordered_set<K> reject_set(keys_D.begin(), keys_D.end());

  // 按 key 来源分组统计
  size_t evicted_old_cnt = 0;
  size_t rejected_new_cnt = 0;
  for (size_t i = 0; i < mixed_evict.counter; i++) {
    const K k = mixed_evict.keys[i];
    if (bucket0_set.find(k) != bucket0_set.end()) {
      evicted_old_cnt++;
    } else if (reject_set.find(k) != reject_set.end()) {
      rejected_new_cnt++;
    } else {
      FAIL() << "Unexpected evicted key: " << k;
    }
  }
  EXPECT_EQ(evicted_old_cnt, evict_num);
  EXPECT_EQ(rejected_new_cnt, reject_num);

  // find 校验：
  // A：更新后的存在且值/分数正确
  VerifyInsertByFind(table, keys_A, values_A_new, scores_A_new, update_num, dim, update_num);
  // B：插入后的存在且值/分数正确
  VerifyInsertByFind(table, keys_B, values_B, scores_B, insert_num, dim, insert_num);
  // C：新 key_C 应存在
  VerifyInsertByFind(table, keys_C, values_C, scores_C, evict_num, dim, evict_num);
  // D：被拒绝的新 key_D 不存在
  VerifyInsertByFind(table, keys_D, values_D, scores_D, reject_num, dim, 0);

  // bucket0：被淘汰的旧 key 不应再存在；其余旧 key 仍存在（通过统计 found 数量验证）
  DeviceMemory find_mem;
  find_mem.AllocateForFind(bucket_size, dim);
  ASSERT_EQ(aclrtMemcpy(find_mem.keys, bucket_size * sizeof(K),
                        bucket0_keys.data(), bucket_size * sizeof(K),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  table.find(bucket_size, find_mem.keys, find_mem.values_ptr, find_mem.found,
             find_mem.scores, find_mem.stream, true);
  ASSERT_EQ(aclrtSynchronizeStream(find_mem.stream), ACL_ERROR_NONE);
  vector<uint8_t> host_found(bucket_size);
  ASSERT_EQ(aclrtMemcpy(host_found.data(), bucket_size * sizeof(bool),
                        find_mem.found, bucket_size * sizeof(bool),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  size_t found_cnt = 0;
  for (size_t i = 0; i < bucket_size; i++) {
    if (host_found[i]) found_cnt++;
  }
  EXPECT_EQ(found_cnt, bucket_size - evict_num);
}

// 测试用例9: evicted_scores 为 nullptr
TEST_F(InsertAndEvictTest, EvictedScoresNullptr_ShouldWork) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t bucket_size = 128;
  constexpr size_t bucket_idx = 0;
  constexpr size_t init_capacity = bucket_size * 4;

  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES, bucket_size);

  // 先填满桶（低分）
  vector<K> keys_low(bucket_size);
  vector<V> values_low(bucket_size * dim);
  vector<S> scores_low(bucket_size);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      keys_low.data(), scores_low.data(), values_low.data(),
      bucket_size, init_capacity, bucket_size, bucket_idx,
      1, 10000000, 1000);
  for (size_t i = 0; i < bucket_size; i++) {
    scores_low[i] = i;  // [0,127]
  }

  DeviceMemory mem1;
  mem1.AllocateForInsert(bucket_size, dim);
  CopyToDevice(keys_low, values_low, scores_low, mem1, bucket_size, dim);
  table.insert_and_evict(bucket_size, mem1.keys, mem1.values, mem1.scores,
                         mem1.evicted_keys, mem1.evicted_values,
                         /*evicted_scores*/ nullptr,
                         mem1.evicted_counter, mem1.stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(mem1.stream), ACL_ERROR_NONE);
  EXPECT_EQ(GetEvictResult(mem1, bucket_size, dim).counter, 0);

  // 插入高分新 key，触发淘汰 old key；但 evicted_scores 为空指针
  constexpr size_t new_key_num = 32;
  vector<K> keys_high(new_key_num);
  vector<V> values_high(new_key_num * dim);
  vector<S> scores_high(new_key_num);
  create_keys_in_one_buckets_lfu<K, S, V, DEFAULT_DIM>(
      keys_high.data(), scores_high.data(), values_high.data(),
      new_key_num, init_capacity, bucket_size, bucket_idx,
      100000000, 200000000, 1000);
  for (size_t i = 0; i < new_key_num; i++) {
    scores_high[i] = 500 + i;  // > 127，保证准入并淘汰
  }

  DeviceMemory mem2;
  mem2.AllocateForInsert(new_key_num, dim);
  CopyToDevice(keys_high, values_high, scores_high, mem2, new_key_num, dim);
  table.insert_and_evict(new_key_num, mem2.keys, mem2.values, mem2.scores,
                         mem2.evicted_keys, mem2.evicted_values,
                         /*evicted_scores*/ nullptr,
                         mem2.evicted_counter, mem2.stream, true, false);
  ASSERT_EQ(aclrtSynchronizeStream(mem2.stream), ACL_ERROR_NONE);

  auto evict_res = GetEvictResult(mem2, new_key_num, dim);
  EXPECT_EQ(evict_res.counter, new_key_num);

  // evicted_keys 必须来自第一批
  unordered_set<K> low_set(keys_low.begin(), keys_low.end());
  for (size_t i = 0; i < evict_res.counter; i++) {
    EXPECT_TRUE(low_set.find(evict_res.keys[i]) != low_set.end());
  }

  // evicted_values 必须等于被淘汰 old key 的 values（按 key 映射验证，不依赖输出顺序）
  unordered_map<K, vector<V>> low_values_map;
  for (size_t i = 0; i < bucket_size; i++) {
    low_values_map[keys_low[i]] = vector<V>(values_low.begin() + i * dim,
                                            values_low.begin() + (i + 1) * dim);
  }
  for (size_t i = 0; i < evict_res.counter; i++) {
    const K k = evict_res.keys[i];
    auto it = low_values_map.find(k);
    ASSERT_TRUE(it != low_values_map.end());
    vector<V> actual(evict_res.values.begin() + i * dim,
                     evict_res.values.begin() + (i + 1) * dim);
    EXPECT_EQ(it->second, actual);
  }

  // 新 key 全部在表中
  VerifyInsertByFind(table, keys_high, values_high, scores_high, new_key_num, dim, new_key_num);
}

 // 测试用例10: 大规模淘汰测试
 TEST_F(InsertAndEvictTest, MassiveInsertWithEvict) {
  constexpr size_t dim = 8;
  constexpr size_t capacity = 128UL;  // 小容量
  constexpr size_t key_num = 1024;    // 大量插入

  // 1. 初始化哈希表
  HashTable<K, V, S, EVICT_STRATEGY> table;
  InitTable(table, dim, capacity, DEFAULT_HBM_FOR_VALUES);
  EXPECT_EQ(table.size(), 0);

  // 2. 准备测试数据
  vector<K> keys(key_num);
  vector<V> values(key_num * dim);
  vector<S> scores(key_num);

  // 生成连续的 key 和递增的 score（score 越大越不容易被淘汰）
  create_continuous_keys<K, S, V, dim>(keys.data(), scores.data(),
                                       values.data(), key_num, 0);

  // 3. 分配设备内存
  DeviceMemory dev_mem;
  dev_mem.AllocateForInsert(key_num, dim);

  // 4. 拷贝数据到设备
  ASSERT_EQ(aclrtMemcpy(dev_mem.keys, key_num * sizeof(K), keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(dev_mem.values, key_num * dim * sizeof(V), values.data(),
                        key_num * dim * sizeof(V), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(dev_mem.scores, key_num * sizeof(S), scores.data(),
                        key_num * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 5. 执行 insert_and_evict
  table.insert_and_evict(key_num, dev_mem.keys, dev_mem.values, dev_mem.scores,
                         dev_mem.evicted_keys, dev_mem.evicted_values,
                         dev_mem.evicted_scores, dev_mem.evicted_counter,
                         dev_mem.stream);
  ASSERT_EQ(aclrtSynchronizeStream(dev_mem.stream), ACL_ERROR_NONE);

  // 6. 验证表大小 = capacity（表已满）
  EXPECT_EQ(table.size(dev_mem.stream), capacity);

  // 7. 验证淘汰数量 = key_num - capacity
  uint64_t evicted_count = 0;
  ASSERT_EQ(aclrtMemcpy(&evicted_count, sizeof(uint64_t), dev_mem.evicted_counter,
                        sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  EXPECT_EQ(evicted_count, key_num - capacity)
      << "Expected " << (key_num - capacity) << " evictions, got " << evicted_count;

  // 8. 读取淘汰的 keys
  vector<K> evicted_keys(evicted_count);
  ASSERT_EQ(aclrtMemcpy(evicted_keys.data(), evicted_count * sizeof(K),
                        dev_mem.evicted_keys, evicted_count * sizeof(K),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 9. 创建被淘汰的 key 集合
  unordered_set<K> evicted_set(evicted_keys.begin(), evicted_keys.end());

  // 10. 分配 find 所需的设备内存
  DeviceMemory find_mem;
  find_mem.AllocateForFind(key_num, dim);

  // 11. 拷贝所有原始 keys 用于查找
  ASSERT_EQ(aclrtMemcpy(find_mem.keys, key_num * sizeof(K), keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 12. 执行 find
  table.find(key_num, find_mem.keys, find_mem.values_ptr, find_mem.found,
             nullptr, find_mem.stream);
  ASSERT_EQ(aclrtSynchronizeStream(find_mem.stream), ACL_ERROR_NONE);

  // 13. 读取 find 结果
  vector<uint8_t> found(key_num);
  ASSERT_EQ(aclrtMemcpy(found.data(), key_num * sizeof(bool), find_mem.found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 14. 验证结果
  size_t found_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (evicted_set.find(keys[i]) != evicted_set.end()) {
      // 被淘汰的 key 不应该在表中
      EXPECT_FALSE(found[i]) << "Key " << keys[i] << " was evicted but found in table";
    } else {
      // 未被淘汰的 key 应该在表中
      EXPECT_TRUE(found[i]) << "Key " << keys[i] << " was not evicted but not found in table";
      if (found[i]) {
        found_count++;
      }
    }
  }

  // 15. 验证找到的 key 数量 = capacity
  EXPECT_EQ(found_count, capacity)
      << "Expected " << capacity << " keys in table, found " << found_count;

  // 16. 验证表中的 values 正确性（完整验证所有找到的 key）
  vector<V*> values_ptr(key_num);
  ASSERT_EQ(aclrtMemcpy(values_ptr.data(), key_num * sizeof(V*), find_mem.values_ptr,
                        key_num * sizeof(V*), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 对所有找到的 key 进行完整验证
  vector<V> real_values(dim);
  for (size_t i = 0; i < key_num; i++) {
    if (found[i]) {
      // 验证 values_ptr 不为空
      ASSERT_NE(values_ptr[i], nullptr)
          << "Key " << keys[i] << " found but values_ptr is null";

      // 从设备读取 values
      ASSERT_EQ(aclrtMemcpy(real_values.data(), dim * sizeof(V),
                            values_ptr[i], dim * sizeof(V),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);

      // 验证每个维度的值
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(real_values[j], values[i * dim + j])
            << "Value mismatch for key " << keys[i] << " at dimension " << j;
      }
    } else {
      // 未找到的 key，values_ptr 应该为空
      EXPECT_EQ(values_ptr[i], nullptr)
          << "Key " << keys[i] << " not found but values_ptr is not null";
    }
  }
}