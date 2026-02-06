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

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;
constexpr size_t DIM = 64;

template <typename K, typename V, typename S>
class HashTableTestHelper {
 public:
  using size_type = size_t;

  HashTableTestHelper(size_type dim, size_type max_key_num = 1000)
      : dim_(dim), max_key_num_(max_key_num) {
    init_env();
    allocate_memory(max_key_num);
    NPU_CHECK(aclrtCreateStream(&stream_));
  }

  ~HashTableTestHelper() { cleanup(); }

  HashTableTestHelper(const HashTableTestHelper&) = delete;
  HashTableTestHelper& operator=(const HashTableTestHelper&) = delete;

  template <int Strategy = EvictStrategy::kLru>
  std::unique_ptr<HashTable<K, V, S, Strategy>> create_table(
      size_type init_capacity, size_type max_capacity,
      size_type max_bucket_size = 128) {
    HashTableOptions options;
    options.init_capacity = init_capacity;
    options.max_capacity = max_capacity;
    options.max_hbm_for_vectors = max_capacity * dim_ * sizeof(V);
    options.max_bucket_size = max_bucket_size;
    options.dim = dim_;

    auto table = std::make_unique<HashTable<K, V, S, Strategy>>();
    table->init(options);
    return table;
  }

  void generate_data(size_type key_num, K start_key = 1) {
    ASSERT_LE(key_num, max_key_num_)
        << "Key number " << key_num << " exceeds allocated capacity "
        << max_key_num_;
    create_continuous_keys<K, S, V, DIM>(h_keys_, h_scores_, h_values_,
                                         static_cast<int>(key_num), start_key);
    current_key_num_ = key_num;
  }

  void copy_to_device(size_type key_num) {
    ASSERT_LE(key_num, current_key_num_) << "Key number exceeds generated data";
    NPU_CHECK(aclrtMemcpy(d_keys_, key_num * sizeof(K), h_keys_,
                          key_num * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_scores_, key_num * sizeof(S), h_scores_,
                          key_num * sizeof(S), ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_values_, key_num * dim_ * sizeof(V), h_values_,
                          key_num * dim_ * sizeof(V),
                          ACL_MEMCPY_HOST_TO_DEVICE));
  }

  template <int Strategy>
  void insert_data(HashTable<K, V, S, Strategy>* table, size_type key_num) {
    if constexpr (Strategy == EvictStrategy::kLru ||
                  Strategy == EvictStrategy::kEpochLru) {
      table->insert_or_assign(key_num, d_keys_, d_values_, nullptr, stream_);
    } else {
      table->insert_or_assign(key_num, d_keys_, d_values_, d_scores_, stream_);
    }
    NPU_CHECK(aclrtSynchronizeStream(stream_));
  }

  template <int Strategy>
  void verify_initial_state(HashTable<K, V, S, Strategy>* table,
                            size_type expected_capacity,
                            size_type expected_bucket_size) {
    EXPECT_EQ(table->dim(), dim_);
    EXPECT_EQ(table->max_bucket_size(), expected_bucket_size);
    EXPECT_EQ(table->capacity(), expected_capacity);
    EXPECT_EQ(table->bucket_count(), expected_capacity / expected_bucket_size);
    EXPECT_TRUE(table->empty(stream_));
    NPU_CHECK(aclrtSynchronizeStream(stream_));
    EXPECT_EQ(table->size(stream_), 0);
    NPU_CHECK(aclrtSynchronizeStream(stream_));
    EXPECT_FLOAT_EQ(table->load_factor(stream_), 0.0f);
    NPU_CHECK(aclrtSynchronizeStream(stream_));
  }

  template <int Strategy>
  void verify_after_insertion(HashTable<K, V, S, Strategy>* table,
                              size_type expected_size) {
    if (expected_size > 0) {
      EXPECT_FALSE(table->empty(stream_));
      NPU_CHECK(aclrtSynchronizeStream(stream_));
    }
    size_type actual_size = table->size(stream_);
    NPU_CHECK(aclrtSynchronizeStream(stream_));
    EXPECT_EQ(actual_size, expected_size);
    float expected_lf = static_cast<float>(actual_size) / table->capacity();
    float actual_lf = table->load_factor(stream_);
    NPU_CHECK(aclrtSynchronizeStream(stream_));
    EXPECT_NEAR(actual_lf, expected_lf, 0.001f);
  }

  template <int Strategy>
  void verify_after_clear(HashTable<K, V, S, Strategy>* table,
                          size_type original_capacity) {
    EXPECT_TRUE(table->empty(stream_));
    NPU_CHECK(aclrtSynchronizeStream(stream_));
    EXPECT_EQ(table->size(stream_), 0);
    NPU_CHECK(aclrtSynchronizeStream(stream_));
    EXPECT_FLOAT_EQ(table->load_factor(stream_), 0.0f);
    NPU_CHECK(aclrtSynchronizeStream(stream_));
    EXPECT_EQ(table->capacity(), original_capacity);
    EXPECT_EQ(table->dim(), dim_);
  }

  aclrtStream get_stream() const { return stream_; }
  K* get_d_keys() const { return d_keys_; }
  V* get_d_values() const { return d_values_; }

 private:
  void allocate_memory(size_type key_num) {
    NPU_CHECK(aclrtMallocHost((void**)&h_keys_, key_num * sizeof(K)));
    NPU_CHECK(aclrtMallocHost((void**)&h_scores_, key_num * sizeof(S)));
    NPU_CHECK(aclrtMallocHost((void**)&h_values_, key_num * dim_ * sizeof(V)));
    NPU_CHECK(aclrtMalloc((void**)&d_keys_, key_num * sizeof(K),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    NPU_CHECK(aclrtMalloc((void**)&d_scores_, key_num * sizeof(S),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    NPU_CHECK(aclrtMalloc((void**)&d_values_, key_num * dim_ * sizeof(V),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  }

  void free_memory() {
    if (h_keys_) {
      NPU_CHECK(aclrtFreeHost(h_keys_));
      h_keys_ = nullptr;
    }
    if (h_scores_) {
      NPU_CHECK(aclrtFreeHost(h_scores_));
      h_scores_ = nullptr;
    }
    if (h_values_) {
      NPU_CHECK(aclrtFreeHost(h_values_));
      h_values_ = nullptr;
    }
    if (d_keys_) {
      NPU_CHECK(aclrtFree(d_keys_));
      d_keys_ = nullptr;
    }
    if (d_scores_) {
      NPU_CHECK(aclrtFree(d_scores_));
      d_scores_ = nullptr;
    }
    if (d_values_) {
      NPU_CHECK(aclrtFree(d_values_));
      d_values_ = nullptr;
    }
  }

  void cleanup() {
    if (stream_) {
      NPU_CHECK(aclrtDestroyStream(stream_));
      stream_ = nullptr;
    }
    free_memory();
  }

  size_type dim_;
  size_type max_key_num_;
  size_type current_key_num_ = 0;
  K* h_keys_ = nullptr;
  S* h_scores_ = nullptr;
  V* h_values_ = nullptr;
  K* d_keys_ = nullptr;
  S* d_scores_ = nullptr;
  V* d_values_ = nullptr;
  aclrtStream stream_ = nullptr;
};

// 测试1: 初始状态验证
TEST(BasicPropertiesTest, InitialStateVerification) {
  const size_t INIT_CAPACITY = 1024 * 128;  // 128K
  const size_t MAX_CAPACITY = 1024 * 1024;  // 1M
  const size_t MAX_BUCKET_SIZE = 128;

  HashTableTestHelper<K, V, S> helper(DIM, 10000);

  auto table = helper.create_table<EvictStrategy::kLru>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  helper.verify_initial_state(table.get(), INIT_CAPACITY, MAX_BUCKET_SIZE);
}

// 测试2: 插入后状态变化
TEST(BasicPropertiesTest, StateChangeAfterInsertion) {
  const size_t INIT_CAPACITY = 1024 * 128;  // 128K
  const size_t MAX_CAPACITY = 1024 * 1024;  // 1M
  const size_t MAX_BUCKET_SIZE = 128;
  const size_t KEY_NUM = 10000;

  HashTableTestHelper<K, V, S> helper(DIM, KEY_NUM);

  auto table = helper.create_table<EvictStrategy::kLru>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  // 记录原始容量
  size_t original_capacity = table->capacity();
  size_t original_bucket_size = table->max_bucket_size();
  size_t original_bucket_count = table->bucket_count();

  // 生成并插入数据
  helper.generate_data(KEY_NUM);
  helper.copy_to_device(KEY_NUM);
  helper.insert_data(table.get(), KEY_NUM);

  // 验证插入后的状态
  helper.verify_after_insertion(table.get(), KEY_NUM);

  // 验证容量相关属性未改变（未触发扩容）
  EXPECT_EQ(table->capacity(), original_capacity)
      << "Capacity should not change without resize";
  EXPECT_EQ(table->max_bucket_size(), original_bucket_size)
      << "Bucket size should not change";
  EXPECT_EQ(table->bucket_count(), original_bucket_count)
      << "Bucket count should not change without resize";
}

// 测试3: clear操作验证
TEST(BasicPropertiesTest, ClearOperation) {
  const size_t INIT_CAPACITY = 1024 * 128;
  const size_t MAX_CAPACITY = 1024 * 1024;
  const size_t MAX_BUCKET_SIZE = 128;
  const size_t KEY_NUM = 5000;

  HashTableTestHelper<K, V, S> helper(DIM, KEY_NUM * 2);

  auto table = helper.create_table<EvictStrategy::kLru>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  // 插入数据
  helper.generate_data(KEY_NUM);
  helper.copy_to_device(KEY_NUM);
  helper.insert_data(table.get(), KEY_NUM);

  // 验证插入成功
  helper.verify_after_insertion(table.get(), KEY_NUM);

  // 记录清空前的状态
  size_t capacity_before = table->capacity();
  size_t dim_before = table->dim();
  size_t bucket_size_before = table->max_bucket_size();
  size_t bucket_count_before = table->bucket_count();

  size_t size_before = table->size(helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
  EXPECT_EQ(size_before, KEY_NUM) << "Size before clear should be " << KEY_NUM;

  // 执行清空
  table->clear(helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  // 验证清空后的状态
  helper.verify_after_clear(table.get(), capacity_before);

  // 验证所有配置属性未改变
  EXPECT_EQ(table->dim(), dim_before) << "Dimension changed after clear";
  EXPECT_EQ(table->max_bucket_size(), bucket_size_before)
      << "Bucket size changed after clear";
  EXPECT_EQ(table->bucket_count(), bucket_count_before)
      << "Bucket count changed after clear";

  // 验证清空后可以继续使用
  const size_t NEW_KEY_NUM = 100;
  helper.generate_data(NEW_KEY_NUM, 100000);  // 使用新的起始键值
  helper.copy_to_device(NEW_KEY_NUM);
  helper.insert_data(table.get(), NEW_KEY_NUM);

  size_t size_after_reinsert = table->size(helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
  EXPECT_EQ(size_after_reinsert, NEW_KEY_NUM)
      << "Table should work after clear";
}

// 测试4: set_max_capacity功能验证
TEST(BasicPropertiesTest, SetMaxCapacity) {
  const size_t INIT_CAPACITY = 1024 * 128;  // 128K
  const size_t MAX_CAPACITY = 1024 * 1024;  // 1M
  const size_t MAX_BUCKET_SIZE = 128;

  HashTableTestHelper<K, V, S> helper(DIM, 1000);

  auto table = helper.create_table<EvictStrategy::kLru>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  // 测试1: 设置更大的max_capacity（合法：2的幂次方）
  EXPECT_NO_THROW({
    table->set_max_capacity(1024 * 1024 * 4);  // 4M
  }) << "Setting valid max_capacity should not throw";

  // 测试2: 设置非2的幂次方（应抛出异常）
  EXPECT_THROW(
      {
        table->set_max_capacity(1024 * 1000);  // 不是2的幂次方
      },
      std::invalid_argument)
      << "Setting non-power-of-2 should throw";

  // 测试3: 设置小于当前容量的值（应被忽略，不报错）
  size_t current_capacity = table->capacity();
  EXPECT_NO_THROW({ table->set_max_capacity(current_capacity / 2); })
      << "Setting smaller max_capacity should not throw";

  // 验证容量没有变小（操作被忽略）
  EXPECT_EQ(table->capacity(), current_capacity)
      << "Capacity should not decrease";

  // 测试4: 边界值测试 - 设置与当前capacity相等的值
  EXPECT_NO_THROW({ table->set_max_capacity(current_capacity); })
      << "Setting equal max_capacity should not throw";
}

// 测试5: set_global_epoch功能验证
// EpochLru策略：score高32位为global_epoch，低32位为timestamp
TEST(BasicPropertiesTest, SetGlobalEpoch) {
  const size_t INIT_CAPACITY = 1024 * 128;
  const size_t MAX_CAPACITY = 1024 * 1024;
  const size_t MAX_BUCKET_SIZE = 128;
  const size_t KEY_NUM = 1000;

  HashTableTestHelper<K, V, S> helper(DIM, KEY_NUM * 2);

  auto table = helper.create_table<EvictStrategy::kEpochLfu>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  // 测试1: 设置epoch=100并插入数据
  constexpr uint64_t epoch1 = 100;
  table->set_global_epoch(epoch1);

  helper.generate_data(KEY_NUM, 1);
  helper.copy_to_device(KEY_NUM);
  helper.insert_data(table.get(), KEY_NUM);

  EXPECT_EQ(table->size(helper.get_stream()), KEY_NUM);
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  // 分配内存用于find操作（使用value指针数组）
  V** d_values_ptr = nullptr;
  S* d_scores_out = nullptr;
  bool* d_founds = nullptr;
  S* h_scores_out = nullptr;

  NPU_CHECK(aclrtMalloc((void**)&d_values_ptr, KEY_NUM * sizeof(V*),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_scores_out, KEY_NUM * sizeof(S),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_founds, KEY_NUM * sizeof(bool),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMallocHost((void**)&h_scores_out, KEY_NUM * sizeof(S)));

  // 查询并验证score的高32位为epoch1
  bool* h_founds = nullptr;
  NPU_CHECK(aclrtMallocHost((void**)&h_founds, KEY_NUM * sizeof(bool)));

  table->find(KEY_NUM, helper.get_d_keys(), d_values_ptr, d_founds,
              d_scores_out, helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));  // 先等待find完成

  NPU_CHECK(aclrtMemcpy(h_scores_out, KEY_NUM * sizeof(S), d_scores_out,
                        KEY_NUM * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtMemcpy(h_founds, KEY_NUM * sizeof(bool), d_founds,
                        KEY_NUM * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < KEY_NUM; i++) {
    EXPECT_TRUE(h_founds[i]);
    uint64_t high_32bits = (h_scores_out[i] >> 32);
    EXPECT_EQ(high_32bits, epoch1);
  }

  NPU_CHECK(aclrtFreeHost(h_founds));

  // 测试2: 更新epoch=101并插入新数据
  constexpr uint64_t epoch2 = 101;
  table->set_global_epoch(epoch2);

  helper.generate_data(KEY_NUM, KEY_NUM + 1);
  helper.copy_to_device(KEY_NUM);
  helper.insert_data(table.get(), KEY_NUM);

  EXPECT_EQ(table->size(helper.get_stream()), KEY_NUM * 2);
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  // 验证第二批数据score的高32位为epoch2
  NPU_CHECK(aclrtMallocHost((void**)&h_founds, KEY_NUM * sizeof(bool)));

  table->find(KEY_NUM, helper.get_d_keys(), d_values_ptr, d_founds,
              d_scores_out, helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  NPU_CHECK(aclrtMemcpy(h_scores_out, KEY_NUM * sizeof(S), d_scores_out,
                        KEY_NUM * sizeof(S), ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtMemcpy(h_founds, KEY_NUM * sizeof(bool), d_founds,
                        KEY_NUM * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < KEY_NUM; i++) {
    EXPECT_TRUE(h_founds[i]);
    uint64_t high_32bits = (h_scores_out[i] >> 32);
    EXPECT_EQ(high_32bits, epoch2);
  }

  NPU_CHECK(aclrtFreeHost(h_founds));

  NPU_CHECK(aclrtFree(d_values_ptr));
  NPU_CHECK(aclrtFree(d_scores_out));
  NPU_CHECK(aclrtFree(d_founds));
  NPU_CHECK(aclrtFreeHost(h_scores_out));
}

// 测试6: 空哈希表操作
TEST(BasicPropertiesTest, EmptyTableOperations) {
  const size_t INIT_CAPACITY = 1024 * 128;
  const size_t MAX_CAPACITY = 1024 * 1024;
  const size_t MAX_BUCKET_SIZE = 128;

  HashTableTestHelper<K, V, S> helper(DIM, 1000);

  auto table = helper.create_table<EvictStrategy::kLru>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  // 验证空表状态
  EXPECT_TRUE(table->empty(helper.get_stream())) << "Empty table check failed";
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  EXPECT_EQ(table->size(helper.get_stream()), 0) << "Size should be 0";
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  EXPECT_FLOAT_EQ(table->load_factor(helper.get_stream()), 0.0f)
      << "Load factor should be 0.0";
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  // 对空表执行clear（应该安全）
  EXPECT_NO_THROW({
    table->clear(helper.get_stream());
    NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
  }) << "Clear on empty table should not throw";

  // 清空后仍然为空
  EXPECT_TRUE(table->empty(helper.get_stream()))
      << "Table should still be empty after clear";
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
}

// 测试7: 高负载因子测试
TEST(BasicPropertiesTest, HighLoadFactor) {
  const size_t INIT_CAPACITY = 1024;
  const size_t MAX_CAPACITY = 1024;
  const size_t MAX_BUCKET_SIZE = 128;
  const float TARGET_LOAD_FACTOR = 0.95f;  // 目标负载率
  const float EPSILON = 0.001f;            // 精度阈值

  HashTableTestHelper<K, V, S> helper(DIM, 2000);

  auto table = helper.create_table<EvictStrategy::kLru>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  // Step 1: 初始插入（接近目标负载率）
  size_t key_num_init = static_cast<size_t>(INIT_CAPACITY * TARGET_LOAD_FACTOR);
  K start = 1;
  helper.generate_data(key_num_init, start);
  helper.copy_to_device(key_num_init);
  helper.insert_data(table.get(), key_num_init);
  start += key_num_init;

  // Step 2: 精确控制负载率
  float real_load_factor = table->load_factor(helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));

  while (TARGET_LOAD_FACTOR - real_load_factor > EPSILON) {
    size_t key_num_append = static_cast<size_t>(
        (TARGET_LOAD_FACTOR - real_load_factor) * INIT_CAPACITY);

    if (key_num_append == 0) break;
    key_num_append =
        std::min(key_num_append, size_t(100));  // 每次最多插入100个

    helper.generate_data(key_num_append, start);
    helper.copy_to_device(key_num_append);
    helper.insert_data(table.get(), key_num_append);

    start += key_num_append;
    real_load_factor = table->load_factor(helper.get_stream());
    NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
  }

  // 验证达到目标负载率
  std::cout << "Target load factor: " << TARGET_LOAD_FACTOR
            << ", Actual: " << real_load_factor << std::endl;
  EXPECT_NEAR(real_load_factor, TARGET_LOAD_FACTOR, 0.05f)
      << "Load factor should be close to target";

  // 验证size不超过capacity
  size_t actual_size = table->size(helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
  EXPECT_LE(actual_size, table->capacity())
      << "Size should not exceed capacity";

  // 继续插入更多数据（会触发淘汰）
  const size_t EXTRA_KEY_NUM = 200;
  helper.generate_data(EXTRA_KEY_NUM, 100000);  // 使用不同的键范围
  helper.copy_to_device(EXTRA_KEY_NUM);

  EXPECT_NO_THROW({ helper.insert_data(table.get(), EXTRA_KEY_NUM); })
      << "Insert should work even at high load";

  // 验证size仍不超过capacity（通过淘汰维持）
  size_t size_after = table->size(helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
  EXPECT_LE(size_after, table->capacity())
      << "Size should not exceed capacity after eviction";
}

// 测试8: 容量边界测试
TEST(BasicPropertiesTest, CapacityBoundary) {
  const size_t INIT_CAPACITY = 1024;  // 1K
  const size_t MAX_CAPACITY = 4096;   // 4K (可扩容到4倍)
  const size_t MAX_BUCKET_SIZE = 128;

  HashTableTestHelper<K, V, S> helper(DIM, 10000);

  auto table = helper.create_table<EvictStrategy::kLru>(
      INIT_CAPACITY, MAX_CAPACITY, MAX_BUCKET_SIZE);

  // 验证初始容量
  EXPECT_EQ(table->capacity(), INIT_CAPACITY) << "Initial capacity mismatch";

  // 第一次插入：触发第一次扩容
  const size_t BATCH1 = 900;  // 接近初始容量
  helper.generate_data(BATCH1, 1);
  helper.copy_to_device(BATCH1);
  helper.insert_data(table.get(), BATCH1);

  size_t capacity_after_1 = table->capacity();
  std::cout << "Capacity after batch 1: " << capacity_after_1 << std::endl;

  // 第二次插入：可能触发第二次扩容
  const size_t BATCH2 = 1500;
  helper.generate_data(BATCH2, 10000);
  helper.copy_to_device(BATCH2);
  helper.insert_data(table.get(), BATCH2);

  size_t capacity_after_2 = table->capacity();
  std::cout << "Capacity after batch 2: " << capacity_after_2 << std::endl;

  // 第三次插入：应该接近或达到max_capacity
  const size_t BATCH3 = 2000;
  helper.generate_data(BATCH3, 20000);
  helper.copy_to_device(BATCH3);
  helper.insert_data(table.get(), BATCH3);

  size_t capacity_after_3 = table->capacity();
  std::cout << "Capacity after batch 3: " << capacity_after_3 << std::endl;

  // 验证不超过max_capacity
  EXPECT_LE(capacity_after_3, MAX_CAPACITY)
      << "Capacity should not exceed max_capacity";

  // 验证size不超过capacity
  size_t final_size = table->size(helper.get_stream());
  NPU_CHECK(aclrtSynchronizeStream(helper.get_stream()));
  EXPECT_LE(final_size, table->capacity()) << "Size should not exceed capacity";

  // 第四次插入：验证达到max_capacity后无法继续扩容
  const size_t BATCH4 = 3000;
  helper.generate_data(BATCH4, 30000);
  helper.copy_to_device(BATCH4);
  helper.insert_data(table.get(), BATCH4);

  size_t capacity_after_4 = table->capacity();
  EXPECT_EQ(capacity_after_4, capacity_after_3)
      << "Capacity should not grow beyond max_capacity";
}
