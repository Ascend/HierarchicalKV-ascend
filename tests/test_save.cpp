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
class SaveTest : public ::testing::Test {
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

// 测试1：基本功能测试 - 插入数据后保存
TEST_F(SaveTest, basic_function) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1UL * 1024;

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

  // 插入数据
  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 保存到文件
  MemoryKVFile<K, V, S> file;
  size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);

  // 验证保存的数量
  EXPECT_EQ(saved_count, key_num);
  EXPECT_EQ(file.size(), key_num);
  EXPECT_EQ(file.get_dim(), dim);

  // 验证保存的数据与原数据一致
  const auto& saved_keys = file.get_keys();
  const auto& saved_values = file.get_values();
  const auto& saved_scores = file.get_scores();

  // 创建原始数据的映射
  unordered_map<K, size_t> key_to_idx;
  for (size_t i = 0; i < key_num; i++) {
    key_to_idx[host_keys[i]] = i;
  }

  // 验证每个保存的 key 都存在于原始数据中
  for (size_t i = 0; i < saved_count; i++) {
    ASSERT_TRUE(key_to_idx.count(saved_keys[i]) > 0)
        << "Key " << saved_keys[i] << " not found in original data";

    size_t orig_idx = key_to_idx[saved_keys[i]];
    // 验证 values
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(saved_values[i * dim + j], host_values[orig_idx * dim + j])
          << "Value mismatch at key " << saved_keys[i] << " dim " << j;
    }
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试2：空表测试 - 对空表执行 save 不会崩溃
TEST_F(SaveTest, empty_table) {
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

  // 在空表上执行 save 应该不会崩溃
  MemoryKVFile<K, V, S> file;
  size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(saved_count, 0);
  EXPECT_EQ(file.size(), 0);
}

// 测试3：单个 key 测试 - n=1
TEST_F(SaveTest, single_key) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  K host_key = 12345;
  S host_score = 100;
  vector<V> host_values(dim, 1.5f);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, &host_key, key_num);
  copy_to_device(device_values, host_values.data(), dim);
  copy_to_device(device_scores, &host_score, key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 保存到文件
  MemoryKVFile<K, V, S> file;
  size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(saved_count, key_num);
  EXPECT_EQ(file.size(), key_num);

  // 验证保存的数据
  const auto& saved_keys = file.get_keys();
  const auto& saved_values = file.get_values();

  EXPECT_EQ(saved_keys[0], host_key);
  for (size_t j = 0; j < dim; j++) {
    EXPECT_FLOAT_EQ(saved_values[j], host_values[j]);
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试4：不同维度测试
TEST_F(SaveTest, different_dimensions) {
  constexpr size_t key_num = 512;

  // 测试多种维度
  vector<size_t> dims = {4, 8, 16, 32, 64, 128};

  for (size_t dim : dims) {
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLfu> table;
    table.init(options);

    vector<K> host_keys(key_num, 0);
    vector<V> host_values(key_num * dim, 0);
    vector<S> host_scores(key_num, 0);
    create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                                host_values.data(), key_num);

    K* device_keys = alloc_device_mem<K>(key_num);
    V* device_values = alloc_device_mem<V>(key_num * dim);
    S* device_scores = alloc_device_mem<S>(key_num);

    copy_to_device(device_keys, host_keys.data(), key_num);
    copy_to_device(device_values, host_values.data(), key_num * dim);
    copy_to_device(device_scores, host_scores.data(), key_num);

    table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                           stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
    EXPECT_EQ(table.size(), key_num);

    // 保存到文件
    MemoryKVFile<K, V, S> file;
    size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);

    EXPECT_EQ(saved_count, key_num) << "Failed for dim=" << dim;
    EXPECT_EQ(file.get_dim(), dim) << "Failed for dim=" << dim;

    free_device_mem(device_keys);
    free_device_mem(device_values);
    free_device_mem(device_scores);
  }
}

// 测试5：大规模数据测试
TEST_F(SaveTest, large_scale) {
  constexpr size_t dim = 32;
  constexpr size_t key_num = 64UL * 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

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

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 保存到文件
  MemoryKVFile<K, V, S> file;
  size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(saved_count, key_num);
  EXPECT_EQ(file.size(), key_num);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试6：随机 keys 测试
TEST_F(SaveTest, random_keys) {
  constexpr size_t dim = 16;
  constexpr size_t key_num = 2048;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

  vector<K> host_keys(key_num, 0);
  vector<V> host_values(key_num * dim, 0);
  vector<S> host_scores(key_num, 0);
  create_random_keys<K, S, V>(dim, host_keys.data(), host_scores.data(),
                              host_values.data(), key_num);

  K* device_keys = alloc_device_mem<K>(key_num);
  V* device_values = alloc_device_mem<V>(key_num * dim);
  S* device_scores = alloc_device_mem<S>(key_num);

  copy_to_device(device_keys, host_keys.data(), key_num);
  copy_to_device(device_values, host_values.data(), key_num * dim);
  copy_to_device(device_scores, host_scores.data(), key_num);

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 保存到文件
  MemoryKVFile<K, V, S> file;
  size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);

  EXPECT_EQ(saved_count, key_num);
  EXPECT_EQ(file.size(), key_num);

  // 验证所有保存的 keys 都能在原数据中找到
  unordered_set<K> original_keys(host_keys.begin(), host_keys.end());
  for (const auto& key : file.get_keys()) {
    EXPECT_TRUE(original_keys.count(key) > 0)
        << "Saved key " << key << " not found in original data";
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试7：不同工作空间大小测试
TEST_F(SaveTest, different_workspace_sizes) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

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

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 测试不同的工作空间大小
  vector<size_t> workspace_sizes = {
      1L * 1024,           // 1KB - 小工作空间，需要多次迭代
      64L * 1024,          // 64KB
      256L * 1024,         // 256KB
      1L * 1024 * 1024,    // 1MB
      4L * 1024 * 1024,    // 4MB - 大工作空间
  };

  for (size_t ws_size : workspace_sizes) {
    // 计算元组大小
    size_t tuple_size = sizeof(K) + sizeof(S) + sizeof(V) * dim;
    if (ws_size < tuple_size) {
      continue;  // 跳过太小的工作空间
    }

    MemoryKVFile<K, V, S> file;
    size_t saved_count = table.save(&file, ws_size, stream_);

    EXPECT_EQ(saved_count, key_num)
        << "Failed for workspace_size=" << ws_size;
    EXPECT_EQ(file.size(), key_num)
        << "Failed for workspace_size=" << ws_size;
  }

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试8：不同驱逐策略测试
TEST_F(SaveTest, different_evict_strategies) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 512;

  // LRU 策略
  {
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLru> table;
    table.init(options);

    vector<K> host_keys(key_num, 0);
    vector<V> host_values(key_num * dim, 0);
    create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr,
                                         host_values.data(), key_num);

    K* device_keys = alloc_device_mem<K>(key_num);
    V* device_values = alloc_device_mem<V>(key_num * dim);

    copy_to_device(device_keys, host_keys.data(), key_num);
    copy_to_device(device_values, host_values.data(), key_num * dim);

    table.insert_or_assign(key_num, device_keys, device_values, nullptr,
                           stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
    EXPECT_EQ(table.size(), key_num);

    MemoryKVFile<K, V, S> file;
    size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);
    EXPECT_EQ(saved_count, key_num) << "Failed for kLru strategy";

    free_device_mem(device_keys);
    free_device_mem(device_values);
  }

  // LFU 策略
  {
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kLfu> table;
    table.init(options);

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

    table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                           stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
    EXPECT_EQ(table.size(), key_num);

    MemoryKVFile<K, V, S> file;
    size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);
    EXPECT_EQ(saved_count, key_num) << "Failed for kLfu strategy";

    free_device_mem(device_keys);
    free_device_mem(device_values);
    free_device_mem(device_scores);
  }

  // EpochLfu 策略
  {
    HashTableOptions options{
        .init_capacity = init_capacity,
        .max_capacity = init_capacity,
        .max_hbm_for_vectors = hbm_for_values,
        .dim = dim,
        .io_by_cpu = false,
    };
    HashTable<K, V, S, EvictStrategy::kEpochLfu> table;
    table.init(options);
    table.set_global_epoch(1);

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

    table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                           stream_);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
    EXPECT_EQ(table.size(), key_num);

    MemoryKVFile<K, V, S> file;
    size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);
    EXPECT_EQ(saved_count, key_num) << "Failed for kEpochLfu strategy";

    free_device_mem(device_keys);
    free_device_mem(device_values);
    free_device_mem(device_scores);
  }
}

// 测试9：多次保存测试 - 验证多次调用 save 得到相同结果
TEST_F(SaveTest, multiple_saves) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

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

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 第一次保存
  MemoryKVFile<K, V, S> file1;
  size_t saved_count1 = table.save(&file1, 1L * 1024 * 1024, stream_);

  // 第二次保存
  MemoryKVFile<K, V, S> file2;
  size_t saved_count2 = table.save(&file2, 1L * 1024 * 1024, stream_);

  // 验证两次保存的数量相同
  EXPECT_EQ(saved_count1, saved_count2);
  EXPECT_EQ(file1.size(), file2.size());

  // 验证保存的 keys 集合相同
  unordered_set<K> keys1(file1.get_keys().begin(), file1.get_keys().end());
  unordered_set<K> keys2(file2.get_keys().begin(), file2.get_keys().end());
  EXPECT_EQ(keys1, keys2);

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试10：保存后插入新数据再保存测试
TEST_F(SaveTest, save_after_insert) {
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

  // 第一批数据
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

  // 第一次保存
  MemoryKVFile<K, V, S> file1;
  size_t saved_count1 = table.save(&file1, 1L * 1024 * 1024, stream_);
  EXPECT_EQ(saved_count1, key_num1);

  // 插入第二批数据
  vector<K> host_keys2(key_num2, 0);
  vector<V> host_values2(key_num2 * dim, 0);
  vector<S> host_scores2(key_num2, 0);
  create_continuous_keys<K, S, V, dim>(host_keys2.data(), host_scores2.data(),
                                       host_values2.data(), key_num2,
                                       key_num1 + 1);

  K* device_keys2 = alloc_device_mem<K>(key_num2);
  V* device_values2 = alloc_device_mem<V>(key_num2 * dim);
  S* device_scores2 = alloc_device_mem<S>(key_num2);

  copy_to_device(device_keys2, host_keys2.data(), key_num2);
  copy_to_device(device_values2, host_values2.data(), key_num2 * dim);
  copy_to_device(device_scores2, host_scores2.data(), key_num2);

  table.insert_or_assign(key_num2, device_keys2, device_values2, device_scores2,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num1 + key_num2);

  // 第二次保存
  MemoryKVFile<K, V, S> file2;
  size_t saved_count2 = table.save(&file2, 1L * 1024 * 1024, stream_);
  EXPECT_EQ(saved_count2, key_num1 + key_num2);

  free_device_mem(device_keys1);
  free_device_mem(device_values1);
  free_device_mem(device_scores1);
  free_device_mem(device_keys2);
  free_device_mem(device_values2);
  free_device_mem(device_scores2);
}

// 测试11：LocalKVFile 写入测试 - 使用本地文件系统保存数据
TEST_F(SaveTest, local_kv_file_write_basic) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

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

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
  EXPECT_EQ(table.size(), key_num);

  // 使用 LocalKVFile 保存到本地文件
  const std::string keys_path = "/tmp/test_save_keys.bin";
  const std::string values_path = "/tmp/test_save_values.bin";
  const std::string scores_path = "/tmp/test_save_scores.bin";

  LocalKVFile<K, V, S> file;
  ASSERT_TRUE(file.open(keys_path, values_path, scores_path, "wb"));

  size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);
  file.close();

  EXPECT_EQ(saved_count, key_num);

  // 验证文件已创建且有内容
  std::ifstream keys_file(keys_path, std::ios::binary | std::ios::ate);
  std::ifstream values_file(values_path, std::ios::binary | std::ios::ate);
  std::ifstream scores_file(scores_path, std::ios::binary | std::ios::ate);

  ASSERT_TRUE(keys_file.is_open());
  ASSERT_TRUE(values_file.is_open());
  ASSERT_TRUE(scores_file.is_open());

  EXPECT_EQ(keys_file.tellg(), static_cast<std::streamsize>(key_num * sizeof(K)));
  EXPECT_EQ(values_file.tellg(), static_cast<std::streamsize>(key_num * dim * sizeof(V)));
  EXPECT_EQ(scores_file.tellg(), static_cast<std::streamsize>(key_num * sizeof(S)));

  keys_file.close();
  values_file.close();
  scores_file.close();

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试12：LocalKVFile 写入后读取验证 - 验证写入的数据正确
TEST_F(SaveTest, local_kv_file_write_verify_content) {
  constexpr size_t dim = 8;
  constexpr size_t key_num = 512;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

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

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 LocalKVFile 保存
  const std::string keys_path = "/tmp/test_save_verify_keys.bin";
  const std::string values_path = "/tmp/test_save_verify_values.bin";
  const std::string scores_path = "/tmp/test_save_verify_scores.bin";

  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));

  size_t saved_count = table.save(&write_file, 1L * 1024 * 1024, stream_);
  write_file.close();
  EXPECT_EQ(saved_count, key_num);

  // 使用 LocalKVFile 读取并验证内容
  LocalKVFile<K, V, S> read_file;
  ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

  vector<K> read_keys(key_num);
  vector<V> read_values(key_num * dim);
  vector<S> read_scores(key_num);

  size_t read_count = read_file.read(key_num, dim, read_keys.data(),
                                     read_values.data(), read_scores.data());
  read_file.close();

  EXPECT_EQ(read_count, key_num);

  // 验证读取的数据与原始数据一致
  unordered_map<K, size_t> key_to_idx;
  for (size_t i = 0; i < key_num; i++) {
    key_to_idx[host_keys[i]] = i;
  }

  for (size_t i = 0; i < read_count; i++) {
    ASSERT_TRUE(key_to_idx.count(read_keys[i]) > 0)
        << "Key " << read_keys[i] << " not found in original data";

    size_t orig_idx = key_to_idx[read_keys[i]];
    for (size_t j = 0; j < dim; j++) {
      EXPECT_FLOAT_EQ(read_values[i * dim + j], host_values[orig_idx * dim + j])
          << "Value mismatch at key " << read_keys[i] << " dim " << j;
    }
  }

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试13：LocalKVFile 大规模数据写入测试
TEST_F(SaveTest, local_kv_file_write_large_scale) {
  constexpr size_t dim = 32;
  constexpr size_t key_num = 32UL * 1024;

  HashTableOptions options{
      .init_capacity = init_capacity,
      .max_capacity = init_capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
  };
  HashTable<K, V, S, EvictStrategy::kLfu> table;
  table.init(options);

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

  table.insert_or_assign(key_num, device_keys, device_values, device_scores,
                         stream_);
  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);

  // 使用 LocalKVFile 保存大规模数据
  const std::string keys_path = "/tmp/test_save_large_keys.bin";
  const std::string values_path = "/tmp/test_save_large_values.bin";
  const std::string scores_path = "/tmp/test_save_large_scores.bin";

  LocalKVFile<K, V, S> file;
  ASSERT_TRUE(file.open(keys_path, values_path, scores_path, "wb"));

  size_t saved_count = table.save(&file, 1L * 1024 * 1024, stream_);
  file.close();

  EXPECT_EQ(saved_count, key_num);

  // 验证文件大小正确
  std::ifstream keys_file(keys_path, std::ios::binary | std::ios::ate);
  EXPECT_EQ(keys_file.tellg(), static_cast<std::streamsize>(key_num * sizeof(K)));
  keys_file.close();

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());

  free_device_mem(device_keys);
  free_device_mem(device_values);
  free_device_mem(device_scores);
}

// 测试14：LocalKVFile 打开失败测试
TEST_F(SaveTest, local_kv_file_open_failure) {
  // 尝试打开不存在的目录下的文件
  const std::string keys_path = "/nonexistent_dir/keys.bin";
  const std::string values_path = "/nonexistent_dir/values.bin";
  const std::string scores_path = "/nonexistent_dir/scores.bin";

  LocalKVFile<K, V, S> file;
  EXPECT_FALSE(file.open(keys_path, values_path, scores_path, "wb"));
}

// 测试15：LocalKVFile write 接口直接测试
TEST_F(SaveTest, local_kv_file_write_direct) {
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

  const std::string keys_path = "/tmp/test_direct_write_keys.bin";
  const std::string values_path = "/tmp/test_direct_write_values.bin";
  const std::string scores_path = "/tmp/test_direct_write_scores.bin";

  // 使用 write 接口写入
  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));

  size_t written = write_file.write(n, dim, keys.data(), values.data(), scores.data());
  write_file.close();

  EXPECT_EQ(written, n);

  // 使用 read 接口读取并验证
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

// 测试16：LocalKVFile 分批写入测试
TEST_F(SaveTest, local_kv_file_write_batched) {
  constexpr size_t dim = 8;
  constexpr size_t batch_size = 100;
  constexpr size_t num_batches = 5;
  constexpr size_t total_n = batch_size * num_batches;

  const std::string keys_path = "/tmp/test_batched_write_keys.bin";
  const std::string values_path = "/tmp/test_batched_write_values.bin";
  const std::string scores_path = "/tmp/test_batched_write_scores.bin";

  // 分批写入
  LocalKVFile<K, V, S> write_file;
  ASSERT_TRUE(write_file.open(keys_path, values_path, scores_path, "wb"));

  size_t total_written = 0;
  for (size_t batch = 0; batch < num_batches; batch++) {
    vector<K> keys(batch_size);
    vector<V> values(batch_size * dim);
    vector<S> scores(batch_size);

    for (size_t i = 0; i < batch_size; i++) {
      size_t global_idx = batch * batch_size + i;
      keys[i] = global_idx + 1;
      scores[i] = global_idx * 10;
      for (size_t j = 0; j < dim; j++) {
        values[i * dim + j] = static_cast<V>(global_idx * dim + j);
      }
    }

    size_t written = write_file.write(batch_size, dim, keys.data(),
                                      values.data(), scores.data());
    EXPECT_EQ(written, batch_size);
    total_written += written;
  }
  write_file.close();

  EXPECT_EQ(total_written, total_n);

  // 验证文件大小
  std::ifstream keys_file(keys_path, std::ios::binary | std::ios::ate);
  EXPECT_EQ(keys_file.tellg(), static_cast<std::streamsize>(total_n * sizeof(K)));
  keys_file.close();

  // 读取并验证
  LocalKVFile<K, V, S> read_file;
  ASSERT_TRUE(read_file.open(keys_path, values_path, scores_path, "rb"));

  vector<K> read_keys(total_n);
  vector<V> read_values(total_n * dim);
  vector<S> read_scores(total_n);

  size_t read_count = read_file.read(total_n, dim, read_keys.data(),
                                     read_values.data(), read_scores.data());
  read_file.close();

  EXPECT_EQ(read_count, total_n);

  // 验证数据正确性
  for (size_t i = 0; i < total_n; i++) {
    EXPECT_EQ(read_keys[i], i + 1);
    EXPECT_EQ(read_scores[i], i * 10);
  }

  // 清理测试文件
  std::remove(keys_path.c_str());
  std::remove(values_path.c_str());
  std::remove(scores_path.c_str());
}
