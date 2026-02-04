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
#include <fstream>
#include <memory>
#include <random>
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

// 测试用简单文件实现
template <class K, class V, class S>
class MemoryKVFile : public BaseKVFile<K, V, S> {
 public:
  MemoryKVFile() = default;
  ~MemoryKVFile() override = default;

  size_t read(const size_t n, const size_t dim, K* keys, V* values,
              S* scores) override {
    size_t count = std::min(n, keys_.size() - read_pos_);
    if (count == 0) return 0;

    for (size_t i = 0; i < count; i++) {
      keys[i] = keys_[read_pos_ + i];
      scores[i] = scores_[read_pos_ + i];
      for (size_t j = 0; j < dim; j++) {
        values[i * dim + j] = values_[(read_pos_ + i) * dim_ + j];
      }
    }
    read_pos_ += count;
    return count;
  }

  size_t write(const size_t n, const size_t dim, const K* keys, const V* values,
               const S* scores) override {
    dim_ = dim;
    for (size_t i = 0; i < n; i++) {
      keys_.push_back(keys[i]);
      scores_.push_back(scores[i]);
      for (size_t j = 0; j < dim; j++) {
        values_.push_back(values[i * dim + j]);
      }
    }
    return n;
  }

  // 直接设置数据（用于测试load）
  void set_data(const vector<K>& keys, const vector<V>& values,
                const vector<S>& scores, size_t dim) {
    keys_ = keys;
    values_ = values;
    scores_ = scores;
    dim_ = dim;
    read_pos_ = 0;
  }

  // 获取保存的数据
  const vector<K>& get_keys() const { return keys_; }
  const vector<V>& get_values() const { return values_; }
  const vector<S>& get_scores() const { return scores_; }
  size_t get_dim() const { return dim_; }
  size_t size() const { return keys_.size(); }

  // 重置读取位置
  void reset_read_pos() { read_pos_ = 0; }

  // 清空数据
  void clear() {
    keys_.clear();
    values_.clear();
    scores_.clear();
    read_pos_ = 0;
  }

 private:
  vector<K> keys_;
  vector<V> values_;
  vector<S> scores_;
  size_t dim_ = 0;
  size_t read_pos_ = 0;
};

// 测试夹具类，用于复用测试初始化和清理逻辑
class LoadTest : public ::testing::Test {
 protected:
  static constexpr size_t hbm_for_values = 1UL << 30;
  static constexpr size_t init_capacity = 128UL * 1024;

  void SetUp() override {
    init_env();

    size_t total_mem = 0;
    size_t free_mem = 0;
    ASSERT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
              ACL_ERROR_NONE);
    ASSERT_GT(free_mem, hbm_for_values)
        << "free HBM is not enough free:" << free_mem
        << " need:" << hbm_for_values;

    ASSERT_EQ(aclrtCreateStream(&stream_), ACL_ERROR_NONE);
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
  }

  // 辅助函数：分配设备内存
  template <typename T>
  T* alloc_device_mem(size_t count) {
    T* ptr = nullptr;
    EXPECT_EQ(aclrtMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    return ptr;
  }

  // 辅助函数：释放设备内存
  template <typename T>
  void free_device_mem(T* ptr) {
    if (ptr != nullptr) {
      EXPECT_EQ(aclrtFree(ptr), ACL_ERROR_NONE);
    }
  }

  // 辅助函数：主机到设备拷贝
  template <typename T>
  void copy_to_device(T* dst, const T* src, size_t count) {
    EXPECT_EQ(aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  // 辅助函数：设备到主机拷贝
  template <typename T>
  void copy_to_host(T* dst, const T* src, size_t count) {
    EXPECT_EQ(aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_ERROR_NONE);
  }

  aclrtStream stream_ = nullptr;
};

// 测试1：基本功能测试 - 从文件加载数据到空表
TEST_F(LoadTest, basic_function) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1UL * 1024;

  // 准备测试数据
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  // 创建内存文件并设置数据
  MemoryKVFile<K, V, S> file;
  file.set_data(host_keys, host_values, host_scores, dim);

  // 创建空表并加载数据
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  // 加载数据
  size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

  // 验证加载的数量
  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(table.size(), key_num);

  // 验证加载的数据 - 通过 find 接口查询
  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  V** device_value_ptrs = alloc_device_mem<V*>(key_num);
  bool* device_founds = alloc_device_mem<bool>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);

  // 使用 V** 版本的 find 接口
  table.find(key_num, device_keys, device_value_ptrs, device_founds, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 read_from_ptr 将指针指向的值拷贝到连续数组
  read_from_ptr(device_value_ptrs, device_values, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> result_values(key_num * dim);
  vector<uint8_t> result_founds(key_num);
  copy_to_host(result_values.data(), device_values, key_num * dim);
  copy_to_host(result_founds.data(), reinterpret_cast<uint8_t*>(device_founds),
               key_num);

  // 验证所有 key 都能找到
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(result_founds[i] != 0)
        << "Key " << host_keys[i] << " not found";
  }

  // 验证 values 正确
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(result_values[i * dim + j], host_values[i * dim + j])
          << "Value mismatch at key " << host_keys[i] << " dim " << j;
    }
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_value_ptrs);
  free_device_mem(device_founds);
}

// 测试2：空文件测试 - 加载空文件不会崩溃
TEST_F(LoadTest, empty_file) {
  constexpr size_t dim = 8;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  // 加载空文件
  MemoryKVFile<K, V, S> file;
  size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(loaded_count, 0);
  EXPECT_EQ(table.size(), 0);
}

// 测试3：单个 key 测试 - n=1
TEST_F(LoadTest, single_key) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1;

  K host_key = 12345;
  S host_score = 100;
  vector<V> host_values(dim, 1.5f);

  // 创建内存文件并设置数据
  MemoryKVFile<K, V, S> file;
  file.set_data({host_key}, host_values, {host_score}, dim);

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  // 加载数据
  size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(table.size(), key_num);

  // 验证加载的数据
  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(dim);
  V** device_value_ptrs = alloc_device_mem<V*>(key_num);
  bool* device_founds = alloc_device_mem<bool>(key_num);

  copy_to_device(device_keys, &host_key, key_num);

  // 使用 V** 版本的 find 接口
  table.find(key_num, device_keys, device_value_ptrs, device_founds, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 read_from_ptr 将指针指向的值拷贝到连续数组
  read_from_ptr(device_value_ptrs, device_values, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> result_values(dim);
  bool result_found;
  copy_to_host(result_values.data(), device_values, dim);
  copy_to_host(&result_found, device_founds, key_num);

  EXPECT_TRUE(result_found);
  for (size_t j = 0; j < dim; j++) {
    EXPECT_FLOAT_EQ(result_values[j], host_values[j]);
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_value_ptrs);
  free_device_mem(device_founds);
}

// 测试4：不同维度测试
TEST_F(LoadTest, different_dimensions) {
  constexpr size_t key_num = 512;

  // 测试多种维度
  vector<size_t> dims = {4, 8, 16, 32, 64, 128};

  for (size_t dim : dims) {
    vector<K> host_keys(key_num, 0);
    vector<V> host_values(key_num * dim, 0);
    vector<S> host_scores(key_num, 0);
    create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                                host_values.data(), key_num);

    // 创建内存文件并设置数据
    MemoryKVFile<K, V, S> file;
    file.set_data(host_keys, host_values, host_scores, dim);

    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLfu> table;
    table.init(options);

    // 加载数据
    size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

    EXPECT_EQ(loaded_count, key_num) << "Failed for dim=" << dim;
    EXPECT_EQ(table.size(), key_num) << "Failed for dim=" << dim;
  }
}

// 测试5：大规模数据测试
TEST_F(LoadTest, large_scale) {
  constexpr size_t dim = 32;
  constexpr size_t key_num = 64UL * 1024;

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  // 创建内存文件并设置数据
  MemoryKVFile<K, V, S> file;
  file.set_data(host_keys, host_values, host_scores, dim);

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  // 加载数据
  size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(table.size(), key_num);
}

// 测试6：随机 keys 测试
TEST_F(LoadTest, random_keys) {
  constexpr size_t dim = 16;
  constexpr size_t key_num = 2048;

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                              host_values.data(), key_num);

  // 创建内存文件并设置数据
  MemoryKVFile<K, V, S> file;
  file.set_data(host_keys, host_values, host_scores, dim);

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  // 加载数据
  size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(table.size(), key_num);

  // 验证所有 key 都能找到
  K* device_keys = alloc_device_mem<K>(key_num);
  bool* device_founds = alloc_device_mem<bool>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  V** device_value_ptrs = alloc_device_mem<V*>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);

  // 使用 V** 版本的 find 接口
  table.find(key_num, device_keys, device_value_ptrs, device_founds, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 read_from_ptr 将指针指向的值拷贝到连续数组
  read_from_ptr(device_value_ptrs, device_values, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<uint8_t> result_founds(key_num);
  copy_to_host(result_founds.data(), reinterpret_cast<uint8_t*>(device_founds),
               key_num);

  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(result_founds[i] != 0)
        << "Key " << host_keys[i] << " not found";
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_value_ptrs);
  free_device_mem(device_founds);
}

// 测试7：不同工作空间大小测试
TEST_F(LoadTest, different_workspace_sizes) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1024;

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  // 测试不同的工作空间大小
  vector<size_t> workspace_sizes = {
    1L * 1024,         // 1KB - 小工作空间，需要多次迭代
    64L * 1024,        // 64KB
    256L * 1024,       // 256KB
    1L * 1024 * 1024,  // 1MB
    4L * 1024 * 1024,  // 4MB - 大工作空间
  };

  for (size_t ws_size : workspace_sizes) {
    // 计算元组大小
    size_t tuple_size = sizeof(K) + sizeof(S) + sizeof(V) * dim;
    if (ws_size < tuple_size) {
      continue;  // 跳过太小的工作空间
    }

    // 创建内存文件并设置数据
    MemoryKVFile<K, V, S> file;
    file.set_data(host_keys, host_values, host_scores, dim);

    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLfu> table;
    table.init(options);

    // 加载数据
    size_t loaded_count = table.load(&file, ws_size, stream_);

    EXPECT_EQ(loaded_count, key_num) << "Failed for workspace_size=" << ws_size;
    EXPECT_EQ(table.size(), key_num) << "Failed for workspace_size=" << ws_size;
  }
}

// 测试8：不同驱逐策略测试
TEST_F(LoadTest, different_evict_strategies) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 512;

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  // LRU 策略
  {
    MemoryKVFile<K, V, S> file;
    file.set_data(host_keys, host_values, host_scores, dim);

    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLru> table;
    table.init(options);

    size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);
    EXPECT_EQ(loaded_count, key_num) << "Failed for kLru strategy";
    EXPECT_EQ(table.size(), key_num) << "Failed for kLru strategy";
  }

  // LFU 策略
  {
    MemoryKVFile<K, V, S> file;
    file.set_data(host_keys, host_values, host_scores, dim);

    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLfu> table;
    table.init(options);

    size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);
    EXPECT_EQ(loaded_count, key_num) << "Failed for kLfu strategy";
    EXPECT_EQ(table.size(), key_num) << "Failed for kLfu strategy";
  }

  // EpochLfu 策略
  {
    MemoryKVFile<K, V, S> file;
    file.set_data(host_keys, host_values, host_scores, dim);

    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kEpochLfu> table;
    table.init(options);

    size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);
    EXPECT_EQ(loaded_count, key_num) << "Failed for kEpochLfu strategy";
    EXPECT_EQ(table.size(), key_num) << "Failed for kEpochLfu strategy";
  }
}

// 测试9：向非空表加载数据测试
TEST_F(LoadTest, load_to_non_empty_table) {
  constexpr size_t dim = 8;
  constexpr size_t key_num1 = 512;
  constexpr size_t key_num2 = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  // 先插入第一批数据
  vector<K> host_keys1(key_num1, 0);
  vector<V> host_values1(key_num1 * dim, 0);
  vector<S> host_scores1(key_num1, 0);
  create_continuous_keys<K, S, V, dim>(host_keys1.data(), host_scores1.data(),
                                       host_values1.data(), key_num1, 1);

  K* device_keys1 = alloc_device_mem<K>(key_num1);
  V* device_values1 = alloc_device_mem<V>(key_num1 * dim);
  S* device_scores1 = alloc_device_mem<S>(key_num1);

  copy_to_device(device_keys1, host_keys1.data(), key_num1);
  copy_to_device(device_values1, host_values1.data(), key_num1 * dim);
  copy_to_device(device_scores1, host_scores1.data(), key_num1);

  table.insert_or_assign(key_num1, device_keys1, device_values1, device_scores1,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num1);

  // 准备第二批数据（不同的keys）
  vector<K> host_keys2(key_num2, 0);
  vector<V> host_values2(key_num2 * dim, 0);
  vector<S> host_scores2(key_num2, 0);
  create_continuous_keys<K, S, V, dim>(host_keys2.data(), host_scores2.data(),
                                       host_values2.data(), key_num2,
                                       key_num1 + 1);

  // 创建内存文件并设置数据
  MemoryKVFile<K, V, S> file;
  file.set_data(host_keys2, host_values2, host_scores2, dim);

  // 加载第二批数据
  size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(loaded_count, key_num2);
  EXPECT_EQ(table.size(), key_num1 + key_num2);

  free_device_mem(device_keys1);
  free_device_mem(device_values1);
  free_device_mem(device_scores1);
}

// 测试10：save 后 load 测试 - 验证数据完整性
TEST_F(LoadTest, save_then_load) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1024;

  // 创建源表并插入数据
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> src_table;
  src_table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  src_table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(src_table.size(), key_num);

  // 保存源表数据到文件
  MemoryKVFile<K, V, S> file;
  size_t saved_count = src_table.save(&file, 1L * 1024 * 1024, stream_);
  EXPECT_EQ(saved_count, key_num);

  // 重置文件读取位置
  file.reset_read_pos();

  // 创建目标表并加载数据
  HashTable<K, V, S, EvictStrategy::kLfu> dst_table;
  dst_table.init(options);

  size_t loaded_count = dst_table.load(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(dst_table.size(), key_num);

  // 验证目标表中的数据与源数据一致
  V* result_device_values = alloc_device_mem<V>(key_num * dim);
  V** device_value_ptrs = alloc_device_mem<V*>(key_num);
  bool* device_founds = alloc_device_mem<bool>(key_num);

  // 使用 V** 版本的 find 接口
  dst_table.find(key_num, device_keys, device_value_ptrs, device_founds,
                 nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 read_from_ptr 将指针指向的值拷贝到连续数组
  read_from_ptr(device_value_ptrs, result_device_values, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> result_values(key_num * dim);
  vector<uint8_t> result_founds(key_num);
  copy_to_host(result_values.data(), result_device_values, key_num * dim);
  copy_to_host(result_founds.data(), reinterpret_cast<uint8_t*>(device_founds),
               key_num);

  // 验证所有 key 都能找到且 values 正确
  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(result_founds[i] != 0)
        << "Key " << host_keys[i] << " not found";
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(result_values[i * dim + j], host_values[i * dim + j])
          << "Value mismatch at key " << host_keys[i] << " dim " << j;
    }
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(result_device_values);
  free_device_mem(device_value_ptrs);
  free_device_mem(device_founds);
}

// 测试11：多次加载测试 - 验证多次调用 load 累加数据
TEST_F(LoadTest, multiple_loads) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  size_t total_loaded = 0;

  // 多次加载不同的数据
  for (int batch = 0; batch < 3; batch++) {
    vector<K> host_keys(key_num, 0);
    vector<V> host_values(key_num * dim, 0);
    vector<S> host_scores(key_num, 0);
    create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                         host_values.data(), key_num,
                                         batch * key_num + 1);

    MemoryKVFile<K, V, S> file;
    file.set_data(host_keys, host_values, host_scores, dim);

    size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);
    EXPECT_EQ(loaded_count, key_num) << "Failed for batch " << batch;
    total_loaded += loaded_count;
  }

  EXPECT_EQ(table.size(), total_loaded);
  EXPECT_EQ(total_loaded, key_num * 3);
}

// 测试12：加载覆盖已存在的 key 测试
TEST_F(LoadTest, load_overwrite_existing_keys) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 256;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  // 准备初始数据
  vector<K> host_keys(key_num, 0);
  vector<V> host_values1(key_num * dim, 1.0f);  // 初始 values 全为 1.0
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       nullptr, key_num);
  // 设置初始 values
  for (size_t i = 0; i < key_num * dim; i++) {
    host_values1[i] = 1.0f;
  }

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values1.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  // 插入初始数据
  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 准备新数据（相同的 keys，不同的 values）
  vector<V> host_values2(key_num * dim, 2.0f);  // 新 values 全为 2.0

  MemoryKVFile<K, V, S> file;
  file.set_data(host_keys, host_values2, host_scores, dim);

  // 加载新数据（应覆盖已存在的 keys）
  size_t loaded_count = table.load(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(table.size(), key_num);  // 大小应保持不变

  // 验证 values 已被更新为新值
  bool* device_founds = alloc_device_mem<bool>(key_num);
  V** device_value_ptrs = alloc_device_mem<V*>(key_num);

  // 使用 V** 版本的 find 接口
  table.find(key_num, device_keys, device_value_ptrs, device_founds, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 read_from_ptr 将指针指向的值拷贝到连续数组
  read_from_ptr(device_value_ptrs, device_values, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> result_values(key_num * dim);
  copy_to_host(result_values.data(), device_values, key_num * dim);

  for (size_t i = 0; i < key_num * dim; i++) {
    EXPECT_FLOAT_EQ(result_values[i], 2.0f)
        << "Value should be updated to 2.0 at index " << i;
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_value_ptrs);
  free_device_mem(device_scores);
  free_device_mem(device_founds);
}

// 测试13：LocalKVFile 读取测试 - 从本地文件加载数据
TEST_F(LoadTest, local_kv_file_read_basic) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1024;

  // 准备测试数据并写入文件
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  const std::string keys_path = "/tmp/test_load_keys.bin";
  const std::string values_path = "/tmp/test_load_values.bin";
  const std::string scores_path = "/tmp/test_load_scores.bin";

  // 先使用 LocalKVFile 写入测试数据
  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));
  size_t written = write_file.write(key_num, dim, host_keys.data(),
                                    host_values.data(), host_scores.data());
  write_file.close();
  ASSERT_EQ(written, key_num);

  // 创建空表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);
  EXPECT_EQ(table.size(), 0);

  // 使用 LocalKVFile 从文件加载数据
  LocalKVFile<K, V, S> read_file;
  ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

  size_t loaded_count = table.load(&read_file, 1L * 1024 * 1024, stream_);
  read_file.close();

  // 验证加载的数量
  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(table.size(), key_num);

  // 验证加载的数据
  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  V** device_value_ptrs = alloc_device_mem<V*>(key_num);
  bool* device_founds = alloc_device_mem<bool>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);

  table.find(key_num, device_keys, device_value_ptrs, device_founds, nullptr,
             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_value_ptrs, device_values, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> result_values(key_num * dim);
  vector<uint8_t> result_founds(key_num);
  copy_to_host(result_values.data(), device_values, key_num * dim);
  copy_to_host(result_founds.data(), reinterpret_cast<uint8_t*>(device_founds),
               key_num);

  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(result_founds[i] != 0)
        << "Key " << host_keys[i] << " not found";
  }

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(result_values[i * dim + j], host_values[i * dim + j])
          << "Value mismatch at key " << host_keys[i] << " dim " << j;
    }
  }

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_value_ptrs);
  free_device_mem(device_founds);
}

// 测试14：LocalKVFile 空文件读取测试
TEST_F(LoadTest, local_kv_file_read_empty) {
  constexpr size_t dim = 8;

  const std::string keys_path = "/tmp/test_load_empty_keys.bin";
  const std::string values_path = "/tmp/test_load_empty_values.bin";
  const std::string scores_path = "/tmp/test_load_empty_scores.bin";

  // 创建空文件
  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));
  write_file.close();

  // 创建空表
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  // 从空文件加载
  LocalKVFile<K, V, S> read_file;
  ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

  size_t loaded_count = table.load(&read_file, 1L * 1024 * 1024, stream_);
  read_file.close();

  EXPECT_EQ(loaded_count, 0);
  EXPECT_EQ(table.size(), 0);

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());
}

// 测试15：LocalKVFile 大规模数据读取测试
TEST_F(LoadTest, local_kv_file_read_large_scale) {
  constexpr size_t dim = 32;
  constexpr size_t key_num = 32UL * 1024;

  // 准备测试数据并写入文件
  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  const std::string keys_path = "/tmp/test_load_large_keys.bin";
  const std::string values_path = "/tmp/test_load_large_values.bin";
  const std::string scores_path = "/tmp/test_load_large_scores.bin";

  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));
  size_t written = write_file.write(key_num, dim, host_keys.data(),
                                    host_values.data(), host_scores.data());
  write_file.close();
  ASSERT_EQ(written, key_num);

  // 创建空表并加载数据
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  LocalKVFile<K, V, S> read_file;
  ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

  size_t loaded_count = table.load(&read_file, 1L * 1024 * 1024, stream_);
  read_file.close();

  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(table.size(), key_num);

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());
}

// 测试16：LocalKVFile read 接口直接测试
TEST_F(LoadTest, local_kv_file_read_direct) {
  constexpr size_t dim = 4;
  constexpr size_t n = 10;

  // 准备测试数据
  vector<K> keys(n);
  vector<V> values(n * dim);
  vector<S> scores(n);

  for (size_t i = 0; i < n; i++) {
    keys[i] = i + 100;
    scores[i] = i * 10;
    for (size_t j = 0; j < dim; j++) {
      values[i * dim + j] = static_cast<V>(i * dim + j) * 0.5f;
    }
  }

  const std::string keys_path = "/tmp/test_direct_read_keys.bin";
  const std::string values_path = "/tmp/test_direct_read_values.bin";
  const std::string scores_path = "/tmp/test_direct_read_scores.bin";

  // 写入测试数据
  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));
  size_t written = write_file.write(n, dim, keys.data(), values.data(),
                                    scores.data());
  write_file.close();
  EXPECT_EQ(written, n);

  // 使用 read 接口读取
  LocalKVFile<K, V, S> read_file;
  ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

  vector<K> read_keys(n);
  vector<V> read_values(n * dim);
  vector<S> read_scores(n);

  size_t read_count = read_file.read(n, dim, read_keys.data(),
                                     read_values.data(), read_scores.data());
  read_file.close();

  EXPECT_EQ(read_count, n);

  // 验证数据一致性
  for (size_t i = 0; i < n; i++) {
    EXPECT_EQ(read_keys[i], keys[i]);
    EXPECT_EQ(read_scores[i], scores[i]);
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(read_values[i * dim + j], values[i * dim + j]);
    }
  }

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());
}

// 测试17：LocalKVFile 分批读取测试
TEST_F(LoadTest, local_kv_file_read_batched) {
  constexpr size_t dim = 8;
  constexpr size_t total_n = 500;
  constexpr size_t batch_size = 100;

  // 准备测试数据
  vector<K> keys(total_n);
  vector<V> values(total_n * dim);
  vector<S> scores(total_n);

  for (size_t i = 0; i < total_n; i++) {
    keys[i] = i + 1;
    scores[i] = i * 10;
    for (size_t j = 0; j < dim; j++) {
      values[i * dim + j] = static_cast<V>(i * dim + j);
    }
  }

  const std::string keys_path = "/tmp/test_batched_read_keys.bin";
  const std::string values_path = "/tmp/test_batched_read_values.bin";
  const std::string scores_path = "/tmp/test_batched_read_scores.bin";

  // 写入测试数据
  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));
  size_t written = write_file.write(total_n, dim, keys.data(), values.data(),
                                    scores.data());
  write_file.close();
  ASSERT_EQ(written, total_n);

  // 分批读取
  LocalKVFile<K, V, S> read_file;
  ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

  size_t total_read = 0;
  size_t batch_idx = 0;

  while (total_read < total_n) {
    vector<K> batch_keys(batch_size);
    vector<V> batch_values(batch_size * dim);
    vector<S> batch_scores(batch_size);

    size_t read_count = read_file.read(batch_size, dim, batch_keys.data(),
                                       batch_values.data(), batch_scores.data());

    if (read_count == 0) break;

    // 验证批次数据
    for (size_t i = 0; i < read_count; i++) {
      size_t global_idx = batch_idx * batch_size + i;
      EXPECT_EQ(batch_keys[i], global_idx + 1);
      EXPECT_EQ(batch_scores[i], global_idx * 10);
      for (size_t j = 0; j < dim; j++) {
        EXPECT_FLOAT_EQ(batch_values[i * dim + j],
                        static_cast<V>(global_idx * dim + j));
      }
    }

    total_read += read_count;
    batch_idx++;
  }
  read_file.close();

  EXPECT_EQ(total_read, total_n);

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());
}

// 测试18：LocalKVFile 保存后加载完整性测试 - 使用本地文件系统
TEST_F(LoadTest, local_kv_file_save_then_load) {
  constexpr size_t dim = 16;
  constexpr size_t key_num = 2048;

  // 创建源表并插入数据
  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> src_table;
  src_table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_continuous_keys<K, S, V, dim>(host_keys.data(), host_scores.data(),
                                       host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  src_table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                             stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(src_table.size(), key_num);

  const std::string keys_path = "/tmp/test_save_load_keys.bin";
  const std::string values_path = "/tmp/test_save_load_values.bin";
  const std::string scores_path = "/tmp/test_save_load_scores.bin";

  // 使用 LocalKVFile 保存数据
  LocalKVFile<K, V, S> save_file;
  ASSERT_TRUE(save_file.open(keys_path, values_path, scores_path, "wb"));

  size_t saved_count = src_table.save(&save_file, 1L * 1024 * 1024, stream_);
  save_file.close();
  EXPECT_EQ(saved_count, key_num);

  // 使用 LocalKVFile 加载数据到新表
  HashTable<K, V, S, EvictStrategy::kLfu> dst_table;
  dst_table.init(options);

  LocalKVFile<K, V, S> load_file;
  ASSERT_TRUE(load_file.open(keys_path, values_path, scores_path, "rb"));

  size_t loaded_count = dst_table.load(&load_file, 1L * 1024 * 1024, stream_);
  load_file.close();

  EXPECT_EQ(loaded_count, key_num);
  EXPECT_EQ(dst_table.size(), key_num);

  // 验证目标表中的数据与源数据一致
  V* result_device_values = alloc_device_mem<V>(key_num * dim);
  V** device_value_ptrs = alloc_device_mem<V*>(key_num);
  bool* device_founds = alloc_device_mem<bool>(key_num);

  dst_table.find(key_num, device_keys, device_value_ptrs, device_founds,
                 nullptr, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  read_from_ptr(device_value_ptrs, result_device_values, dim, key_num, stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  vector<V> result_values(key_num * dim);
  vector<uint8_t> result_founds(key_num);
  copy_to_host(result_values.data(), result_device_values, key_num * dim);
  copy_to_host(result_founds.data(), reinterpret_cast<uint8_t*>(device_founds),
               key_num);

  for (size_t i = 0; i < key_num; i++) {
    EXPECT_TRUE(result_founds[i] != 0)
        << "Key " << host_keys[i] << " not found";
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(result_values[i * dim + j], host_values[i * dim + j])
          << "Value mismatch at key " << host_keys[i] << " dim " << j;
    }
  }

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
  free_device_mem(result_device_values);
  free_device_mem(device_value_ptrs);
  free_device_mem(device_founds);
}

// 测试19：LocalKVFile 打开失败测试
TEST_F(LoadTest, local_kv_file_open_failure) {
  // 尝试打开不存在的文件进行读取
  const std::string keys_path = "/nonexistent_dir/keys.bin";
  const std::string values_path = "/nonexistent_dir/values.bin";
  const std::string scores_path = "/nonexistent_dir/scores.bin";

  LocalKVFile<K, V, S> file;
  EXPECT_FALSE(file.open(keys_path, values_path, scores_path, "rb"));
}

// 测试20：LocalKVFile 不同维度读取测试
TEST_F(LoadTest, local_kv_file_read_different_dimensions) {
  constexpr size_t key_num = 256;

  vector<size_t> dims = {4, 8, 16, 32, 64};

  for (size_t dim : dims) {
    // 准备测试数据
    vector<K> host_keys(key_num, 0);
    vector<V> host_values(key_num * dim, 0);
    vector<S> host_scores(key_num, 0);
    create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                                host_values.data(), key_num);

    const std::string keys_path = "/tmp/test_load_dim_keys.bin";
    const std::string values_path = "/tmp/test_load_dim_values.bin";
    const std::string scores_path = "/tmp/test_load_dim_scores.bin";

    // 写入数据
    LocalKVFile<K, V, S> write_file;
    ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));
    write_file.write(key_num, dim, host_keys.data(), host_values.data(),
                     host_scores.data());
    write_file.close();

    // 创建表并加载
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLfu> table;
    table.init(options);

    LocalKVFile<K, V, S> read_file;
    ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

    size_t loaded_count = table.load(&read_file, 1L * 1024 * 1024, stream_);
    read_file.close();

    EXPECT_EQ(loaded_count, key_num) << "Failed for dim=" << dim;
    EXPECT_EQ(table.size(), key_num) << "Failed for dim=" << dim;

    // 清理测试文件
    std::remove(keys_path.c_str());
    std::remove(values_path.c_str());
    std::remove(scores_path.c_str());
  }
}
