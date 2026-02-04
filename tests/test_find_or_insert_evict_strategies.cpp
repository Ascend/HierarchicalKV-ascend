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
#include <vector>
#include <chrono>
#include <thread>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

// 测试辅助函数：通用的 find_or_insert 测试逻辑
template<typename Table>
void test_find_or_insert_with_strategy(const char* strategy_name) {
  // 1. 初始化
  init_env();

  size_t free_mem = 0;
  size_t total_mem = 0;
  constexpr size_t hbm_for_values = 1UL << 30;
  ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  ASSERT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem << " need:" << hbm_for_values;

  constexpr size_t dim = 8;
  constexpr size_t init_capacity = 128UL * 1024;
  constexpr size_t key_num = 1UL * 1024;

  using K = uint64_t;
  using V = float;
  using S = uint64_t;
  size_t each_key_size = sizeof(K);
  size_t each_value_size = sizeof(V);

  // 2. 建表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };

  Table table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  // 2.1 对于 Epoch 相关策略，设置 global_epoch
  constexpr uint64_t test_epoch = 123;
  if (table.evict_strategy == EvictStrategy::kEpochLru || 
      table.evict_strategy == EvictStrategy::kEpochLfu) {
    table.set_global_epoch(test_epoch);
  }

  // 3. 数据准备
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);

  // 3.2 申请hbm内存
  K* device_keys = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  S* device_scores = nullptr;

  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * each_key_size, ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                        key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 4. 第一次插入
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(), 
                                       host_values.data(), key_num);
  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * each_key_size, host_keys.data(),
                        key_num * each_key_size, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 对于 LFU 和 EpochLFU 策略，需要传递 scores
  S* scores_param = nullptr;
  if (table.evict_strategy == EvictStrategy::kLfu || 
      table.evict_strategy == EvictStrategy::kEpochLfu ||
      table.evict_strategy == EvictStrategy::kCustomized) {
    ASSERT_EQ(aclrtMemcpy(device_scores, key_num * sizeof(S), host_scores.data(),
                          key_num * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
    scores_param = device_scores;
  }

  // 4.2 下发算子（第一次插入）
  table.find_or_insert(key_num, device_keys, device_values_ptr, device_found,
                       scores_param, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);

  // 4.2.1 对于 LRU/EpochLRU 策略，读取第一次插入后的 scores（用于后续验证时钟递增）
  S* device_scores_first = nullptr;
  V** device_values_first = nullptr;
  bool* device_founds_first = nullptr;
  vector<S> host_scores_first;
  
  if (table.evict_strategy == EvictStrategy::kLru || 
      table.evict_strategy == EvictStrategy::kEpochLru) {
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_first),
                          key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_first),
                          key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_founds_first),
                          key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    
    table.find(key_num, device_keys, device_values_first, device_founds_first,
               device_scores_first, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
    
    host_scores_first.resize(key_num, 0);
    ASSERT_EQ(aclrtMemcpy(host_scores_first.data(), key_num * sizeof(S),
                          device_scores_first, key_num * sizeof(S),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
  }

  // 4.3 检查找到的key并写值
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  size_t found_num = 0;
  size_t refused_num = 0;
  size_t insert_num = 0;
  ASSERT_EQ(aclrtMemcpy(host_found, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  vector<void*> expect_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(expect_values_ptr.data(), key_num * sizeof(void*),
                        device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      ASSERT_NE(expect_values_ptr[i], nullptr);
      vector<V> expect_values(dim, i);
      ASSERT_EQ(aclrtMemcpy(expect_values_ptr[i], dim * each_value_size,
                            expect_values.data(), dim * each_value_size,
                            ACL_MEMCPY_HOST_TO_DEVICE),
                ACL_ERROR_NONE);
      found_num++;
    } else {
      if (expect_values_ptr[i] == nullptr) {
        refused_num++;
      } else {
        vector<V> expect_values(dim, i);
        ASSERT_EQ(aclrtMemcpy(expect_values_ptr[i], dim * each_value_size,
                              expect_values.data(), dim * each_value_size,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  ACL_ERROR_NONE);
        insert_num++;
      }
    }
  }
  EXPECT_EQ(found_num, 0);
  EXPECT_EQ(refused_num, 0);
  EXPECT_EQ(insert_num, key_num);

  // 5. 第二次查找，验证结果（会触发 LRU/EpochLRU 的时钟更新）
  ASSERT_EQ(aclrtMemset(device_values_ptr, key_num * sizeof(V*), 0, 
                        key_num * sizeof(V*)), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemset(device_found, key_num * sizeof(bool), 0, 
                        key_num * sizeof(bool)), ACL_ERROR_NONE);

  // 添加延迟确保时钟增加（对于LRU/EpochLRU）
  if (table.evict_strategy == EvictStrategy::kLru || 
      table.evict_strategy == EvictStrategy::kEpochLru) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  table.find_or_insert(key_num, device_keys, device_values_ptr, device_found,
                       scores_param, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(stream), key_num);

  vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*), 
                        device_values_ptr, key_num * sizeof(void*), 
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 5.1 验证指针相同
  EXPECT_EQ(real_values_ptr, expect_values_ptr);

  bool* host_found_again = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found_again),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_found_again, key_num * sizeof(bool), device_found,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 5.2 验证值相同
  size_t verify_success = 0;
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_NE(host_found[i], host_found_again[i]);
    if (host_found_again[i]) {
      ASSERT_NE(real_values_ptr[i], nullptr);
      vector<V> expect_values(dim, i);
      vector<V> real_values(dim, 0);
      ASSERT_EQ(aclrtMemcpy(real_values.data(), dim * each_value_size, 
                            real_values_ptr[i], dim * each_value_size, 
                            ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_ERROR_NONE);
      EXPECT_EQ(real_values, expect_values);
      verify_success++;
    } else {
      EXPECT_EQ(real_values_ptr[i], nullptr);
    }
  }
  EXPECT_EQ(verify_success, key_num);

  // 6. 验证策略相关的 scores 是否正确更新
  S* device_scores_output = nullptr;
  V** device_values_for_find = nullptr;
  bool* device_founds_for_find = nullptr;
  
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores_output),
                        key_num * sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_for_find),
                        key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_founds_for_find),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  
  // 使用 find 读取 scores
  table.find(key_num, device_keys, device_values_for_find, device_founds_for_find,
             device_scores_output, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
  
  // 将 scores 拷贝回 host
  vector<S> host_scores_actual(key_num, 0);
  ASSERT_EQ(aclrtMemcpy(host_scores_actual.data(), key_num * sizeof(S),
                        device_scores_output, key_num * sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  
  // 验证 scores
  for (size_t i = 0; i < key_num; i++) {
    if (!host_found_again[i]) continue;
    
    S actual_score = host_scores_actual[i];
    
    switch (table.evict_strategy) {
      case EvictStrategy::kLru: {
        S first_score = host_scores_first[i];
        EXPECT_GT(actual_score, 0);
        EXPECT_GT(actual_score, first_score);
        break;
      }
      
      case EvictStrategy::kLfu: {
        S expected_score = 2 * host_scores[i];
        EXPECT_EQ(actual_score, expected_score);
        break;
      }
      
      case EvictStrategy::kEpochLru: {
        S first_score = host_scores_first[i];
        S first_cycle_part = first_score & 0xFFFFFFFF;
        S epoch_part = actual_score >> 32;
        S cycle_part = actual_score & 0xFFFFFFFF;
        constexpr uint64_t test_epoch = 123;
        EXPECT_EQ(epoch_part, test_epoch);
        EXPECT_GT(cycle_part, 0);
        EXPECT_GT(cycle_part, first_cycle_part);
        break;
      }
      
      case EvictStrategy::kEpochLfu: {
        S epoch_part = actual_score >> 32;
        S freq_part = actual_score & 0xFFFFFFFF;
        S expected_freq = 2 * (host_scores[i] & 0xFFFFFFFF);
        constexpr uint64_t test_epoch = 123;
        EXPECT_EQ(epoch_part, test_epoch);
        EXPECT_EQ(freq_part, expected_freq);
        break;
      }
      
      case EvictStrategy::kCustomized: {
        S expected_score = host_scores[i];
        EXPECT_EQ(actual_score, expected_score);
        break;
      }
      
      default:
        FAIL();
    }
  }
  
  // 验证策略参数正确传递
  switch (table.evict_strategy) {
    case EvictStrategy::kLru:
    case EvictStrategy::kEpochLru:
      EXPECT_EQ(scores_param, nullptr);
      break;
    case EvictStrategy::kLfu:
    case EvictStrategy::kEpochLfu:
    case EvictStrategy::kCustomized:
      EXPECT_NE(scores_param, nullptr);
      break;
  }
  
  ASSERT_EQ(aclrtFree(device_scores_output), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_for_find), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_founds_for_find), ACL_ERROR_NONE);

  // 释放第一次插入时的临时内存
  if (table.evict_strategy == EvictStrategy::kLru || 
      table.evict_strategy == EvictStrategy::kEpochLru) {
    ASSERT_EQ(aclrtFree(device_scores_first), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_values_first), ACL_ERROR_NONE);
    ASSERT_EQ(aclrtFree(device_founds_first), ACL_ERROR_NONE);
  }

  // 7. 释放内存
  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_found_again), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 测试 kLru 策略
TEST(test_find_or_insert_evict_strategies, kLru) {
  using Table = HashTable<uint64_t, float, uint64_t, EvictStrategy::kLru>;
  test_find_or_insert_with_strategy<Table>("kLru");
}

// 测试 kLfu 策略
TEST(test_find_or_insert_evict_strategies, kLfu) {
  using Table = HashTable<uint64_t, float, uint64_t, EvictStrategy::kLfu>;
  test_find_or_insert_with_strategy<Table>("kLfu");
}

// 测试 kEpochLru 策略
TEST(test_find_or_insert_evict_strategies, kEpochLru) {
  using Table = HashTable<uint64_t, float, uint64_t, EvictStrategy::kEpochLru>;
  test_find_or_insert_with_strategy<Table>("kEpochLru");
}

// 测试 kEpochLfu 策略
TEST(test_find_or_insert_evict_strategies, kEpochLfu) {
  using Table = HashTable<uint64_t, float, uint64_t, EvictStrategy::kEpochLfu>;
  test_find_or_insert_with_strategy<Table>("kEpochLfu");
}

// 测试 kCustomized 策略
TEST(test_find_or_insert_evict_strategies, kCustomized) {
  using Table = HashTable<uint64_t, float, uint64_t, EvictStrategy::kCustomized>;
  test_find_or_insert_with_strategy<Table>("kCustomized");
}
