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
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;

class LockUnlockKeysTest : public ::testing::Test {
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

  // 辅助函数：创建和初始化哈希表
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
};

// 用例1: 边界测试 - 空查询 (n=0)
TEST_F(LockUnlockKeysTest, BoundaryTest_EmptyQuery) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 0;  // 边界值

  HashTable<K, V> table;
  InitTable(table, dim, DEFAULT_INIT_CAPACITY, DEFAULT_HBM_FOR_VALUES);

  K* device_keys = nullptr;
  K** device_locked_ptrs = nullptr;
  bool* device_succeed = nullptr;

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 调用lock_keys，n=0应该直接返回
  table.lock_keys(key_num, device_keys, device_locked_ptrs, device_succeed,
                  stream, nullptr);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 调用unlock_keys，n=0应该直接返回
  table.unlock_keys(key_num, device_locked_ptrs, device_keys, device_succeed,
                    stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);

  // 验证：函数正常返回，无崩溃
  SUCCEED();
}

// 用例2: 基本功能 - 锁定已存在的key
TEST_F(LockUnlockKeysTest, BasicFunction_LockExistingKeys) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 申请设备内存
  K* device_keys = nullptr;
  V* device_values = nullptr;
  K** device_locked_ptrs = nullptr;
  bool* device_succeed = nullptr;

  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_locked_ptrs),
                        key_num * sizeof(K*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_succeed),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num);

  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * dim * sizeof(V),
                        host_values.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入所有键
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 锁定所有key
  table.lock_keys(key_num, device_keys, device_locked_ptrs, device_succeed,
                  stream, nullptr);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证锁定结果
  bool* host_succeed = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_succeed),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_succeed, key_num * sizeof(bool), device_succeed,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 所有key都应该锁定成功
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(host_succeed[i]) << "Key " << i << " lock failed";
  }

  // 验证locked_ptrs不为空
  vector<void*> locked_ptrs(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(locked_ptrs.data(), key_num * sizeof(void*),
                        device_locked_ptrs, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_NE(locked_ptrs[i], nullptr) << "Key " << i << " locked_ptr is null";
  }

  // 解锁所有key
  table.unlock_keys(key_num, device_locked_ptrs, device_keys, device_succeed,
                    stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证解锁结果
  ASSERT_EQ(aclrtMemcpy(host_succeed, key_num * sizeof(bool), device_succeed,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 所有key都应该解锁成功
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(host_succeed[i]) << "Key " << i << " unlock failed";
  }

  // 清理
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_locked_ptrs), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例3: 基本功能 - 锁定不存在的key应该失败
TEST_F(LockUnlockKeysTest, BasicFunction_LockNonExistingKeys) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 申请设备内存
  K* device_keys = nullptr;
  V* device_values = nullptr;
  K** device_locked_ptrs = nullptr;
  bool* device_succeed = nullptr;

  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_locked_ptrs),
                        key_num * sizeof(K*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_succeed),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num);

  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * dim * sizeof(V),
                        host_values.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 注意：这里不插入key，直接尝试锁定不存在的key

  // 尝试锁定不存在的key
  table.lock_keys(key_num, device_keys, device_locked_ptrs, device_succeed,
                  stream, nullptr);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证锁定结果 - 所有key都应该失败
  bool* host_succeed = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_succeed),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_succeed, key_num * sizeof(bool), device_succeed,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 所有key都应该锁定失败
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_FALSE(host_succeed[i]) << "Key " << i << " should lock failed";
  }

  // 验证locked_ptrs应该为空
  vector<void*> locked_ptrs(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(locked_ptrs.data(), key_num * sizeof(void*),
                        device_locked_ptrs, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_EQ(locked_ptrs[i], nullptr) << "Key " << i << " locked_ptr should be null";
  }

  // 清理
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_locked_ptrs), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例4: 基本功能 - 部分key存在，部分不存在
TEST_F(LockUnlockKeysTest, BasicFunction_LockPartialExistingKeys) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 1024;
  constexpr size_t insert_num = 512;  // 只插入一半
  constexpr size_t init_capacity = 128UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 申请设备内存
  K* device_keys = nullptr;
  V* device_values = nullptr;
  K** device_locked_ptrs = nullptr;
  bool* device_succeed = nullptr;

  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_locked_ptrs),
                        key_num * sizeof(K*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_succeed),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num);

  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * dim * sizeof(V),
                        host_values.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 只插入前insert_num个key
  table.insert_or_assign(insert_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 尝试锁定所有key
  table.lock_keys(key_num, device_keys, device_locked_ptrs, device_succeed,
                  stream, nullptr);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证锁定结果
  bool* host_succeed = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_succeed),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_succeed, key_num * sizeof(bool), device_succeed,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  // 前insert_num个应该成功，后面的应该失败
  for (size_t i = 0; i < key_num; i++) {
    if (i < insert_num) {
      EXPECT_TRUE(host_succeed[i]) << "Key " << i << " should lock success";
    } else {
      EXPECT_FALSE(host_succeed[i]) << "Key " << i << " should lock failed";
    }
  }

  // 清理
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_locked_ptrs), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}

// 用例5: 性能测试 - 大规模key锁定
TEST_F(LockUnlockKeysTest, PerformanceTest_LargeScaleKeys) {
  constexpr size_t dim = DEFAULT_DIM;
  constexpr size_t key_num = 65536;  // 64K keys
  constexpr size_t init_capacity = 1024UL * 1024;

  HashTable<K, V> table;
  InitTable(table, dim, init_capacity, DEFAULT_HBM_FOR_VALUES);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);

  // 申请设备内存
  K* device_keys = nullptr;
  V* device_values = nullptr;
  K** device_locked_ptrs = nullptr;
  bool* device_succeed = nullptr;

  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                        key_num * sizeof(K), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values),
                        key_num * dim * sizeof(V), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_locked_ptrs),
                        key_num * sizeof(K*), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_succeed),
                        key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_ERROR_NONE);

  aclrtStream stream = nullptr;
  ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);

  // 生成连续键
  create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                       host_values.data(), key_num);

  ASSERT_EQ(aclrtMemcpy(device_keys, key_num * sizeof(K), host_keys.data(),
                        key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(device_values, key_num * dim * sizeof(V),
                        host_values.data(), key_num * dim * sizeof(V),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_ERROR_NONE);

  // 插入所有键
  table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 锁定所有key
  table.lock_keys(key_num, device_keys, device_locked_ptrs, device_succeed,
                  stream, nullptr);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 验证
  bool* host_succeed = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_succeed),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(aclrtMemcpy(host_succeed, key_num * sizeof(bool), device_succeed,
                        key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);

  size_t success_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    if (host_succeed[i]) success_count++;
  }
  EXPECT_EQ(success_count, key_num);

  // 解锁所有key
  table.unlock_keys(key_num, device_locked_ptrs, device_keys, device_succeed,
                    stream);
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);

  // 清理
  ASSERT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_locked_ptrs), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFree(device_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtFreeHost(host_succeed), ACL_ERROR_NONE);
  ASSERT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
}