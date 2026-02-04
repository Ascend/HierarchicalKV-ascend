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
 #include "acl/acl.h"
 #include "hkv_hashtable.h"
 #include "test_util.h"
 
 using namespace std;
 using namespace npu::hkv;
 using namespace test_util;
 
 using K = uint64_t;
 using V = float;
 using S = uint64_t;
 
 // 测试夹具类，用于复用测试初始化和清理逻辑
 class FindAndUpdateTest : public ::testing::Test {
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
     EXPECT_EQ(aclrtMalloc(reinterpret_cast<void**>(&ptr),
                           count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST),
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
 
 // 测试1：基本功能测试 - 插入后查找所有 keys
 TEST_F(FindAndUpdateTest, basic_function) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 1UL * 1024;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
   EXPECT_EQ(table.size(), 0);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), key_num);
 
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   copy_to_host(host_found.get(), device_found, key_num);
   vector<V*> real_values_ptr(key_num, nullptr);
   copy_to_host(real_values_ptr.data(), device_values_ptr, key_num);
 
   size_t found_num = 0;
   for (size_t i = 0; i < key_num; i++) {
     if (host_found[i]) {
       ASSERT_NE(real_values_ptr[i], nullptr);
       found_num++;
 
       vector<V> real_values(dim, 0);
       copy_to_host(real_values.data(), real_values_ptr[i], dim);
       vector<V> expect_values(host_values.begin() + i * dim,
                               host_values.begin() + i * dim + dim);
       EXPECT_EQ(expect_values, real_values);
     } else {
       EXPECT_EQ(real_values_ptr[i], nullptr);
     }
   }
   EXPECT_EQ(found_num, key_num);
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试2：空表查找 - 查找不存在的 keys，所有 founds 应为 false
 TEST_F(FindAndUpdateTest, empty_table_lookup) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 100;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
   EXPECT_EQ(table.size(), 0);
 
   vector<K> host_keys(key_num, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
 
   // 在空表上查找
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   std::fill_n(host_found.get(), key_num, true);
   copy_to_host(host_found.get(), device_found, key_num);
   vector<V*> real_values_ptr(key_num, reinterpret_cast<V*>(1));  // 非空初始值
   copy_to_host(real_values_ptr.data(), device_values_ptr, key_num);
 
   for (size_t i = 0; i < key_num; i++) {
     EXPECT_FALSE(host_found[i]) << "Key at index " << i << " should not be found";
     EXPECT_EQ(real_values_ptr[i], nullptr) << "Pointer at index " << i << " should be nullptr";
   }
 
   free_device_mem(device_keys);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试3：部分存在测试 - 部分 keys 存在，部分不存在
 TEST_F(FindAndUpdateTest, partial_keys_exist) {
   constexpr size_t dim = 8;
   constexpr size_t insert_key_num = 500;
   constexpr size_t query_key_num = 1000;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   // 插入前 500 个 keys (1-500)
   vector<K> insert_keys(insert_key_num, 0);
   vector<V> insert_values(insert_key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(insert_keys.data(), nullptr, insert_values.data(), insert_key_num, 1);
 
   K* device_insert_keys = alloc_device_mem<K>(insert_key_num);
   V* device_insert_values = alloc_device_mem<V>(insert_key_num * dim);
   copy_to_device(device_insert_keys, insert_keys.data(), insert_key_num);
   copy_to_device(device_insert_values, insert_values.data(), insert_key_num * dim);
 
   table.insert_or_assign(insert_key_num, device_insert_keys, device_insert_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), insert_key_num);
 
   // 查询 1000 个 keys (1-1000)，只有前 500 个存在
   vector<K> query_keys(query_key_num, 0);
   create_continuous_keys<K, S, V, dim>(query_keys.data(), nullptr, nullptr, query_key_num, 1);
 
   K* device_query_keys = alloc_device_mem<K>(query_key_num);
   V** device_values_ptr = alloc_device_mem<V*>(query_key_num);
   bool* device_found = alloc_device_mem<bool>(query_key_num);
 
   copy_to_device(device_query_keys, query_keys.data(), query_key_num);
 
   table.find_and_update(query_key_num, device_query_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[query_key_num]());
   copy_to_host(host_found.get(), device_found, query_key_num);
   vector<V*> real_values_ptr(query_key_num, nullptr);
   copy_to_host(real_values_ptr.data(), device_values_ptr, query_key_num);
 
   size_t found_count = 0;
   size_t not_found_count = 0;
   for (size_t i = 0; i < query_key_num; i++) {
     if (i < insert_key_num) {
       EXPECT_TRUE(host_found[i]) << "Key " << query_keys[i] << " at index " << i << " should be found";
       EXPECT_NE(real_values_ptr[i], nullptr);
       if (host_found[i]) found_count++;
     } else {
       EXPECT_FALSE(host_found[i]) << "Key " << query_keys[i] << " at index " << i << " should not be found";
       EXPECT_EQ(real_values_ptr[i], nullptr);
       if (!host_found[i]) not_found_count++;
     }
   }
   EXPECT_EQ(found_count, insert_key_num);
   EXPECT_EQ(not_found_count, query_key_num - insert_key_num);
 
   free_device_mem(device_insert_keys);
   free_device_mem(device_insert_values);
   free_device_mem(device_query_keys);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试4：边界情况 - n=0 时不崩溃
 TEST_F(FindAndUpdateTest, zero_keys) {
   constexpr size_t dim = 8;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   // n=0 时调用 find_and_update 应该直接返回，不崩溃
   table.find_and_update(0, nullptr, nullptr, nullptr, nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), 0);
 }
 
 // 测试5：单个 key 测试 - n=1
 TEST_F(FindAndUpdateTest, single_key) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 1;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys = {12345};
   vector<V> host_values(dim, 1.5f);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), key_num);
 
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   bool host_found = false;
   copy_to_host(&host_found, device_found, 1);
   EXPECT_TRUE(host_found);
 
   V* real_value_ptr = nullptr;
   copy_to_host(&real_value_ptr, device_values_ptr, 1);
   EXPECT_NE(real_value_ptr, nullptr);
 
   if (real_value_ptr != nullptr) {
     vector<V> real_values(dim, 0);
     copy_to_host(real_values.data(), real_value_ptr, dim);
     EXPECT_EQ(host_values, real_values);
   }
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试6：清表后查找 - 插入、清表、查找
 TEST_F(FindAndUpdateTest, find_after_clear) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 256;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   // 插入数据
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), key_num);
 
   // 清表
   table.clear(stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), 0);
 
   // 清表后查找，所有 keys 应该不存在
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   std::fill_n(host_found.get(), key_num, true);
   copy_to_host(host_found.get(), device_found, key_num);
 
   for (size_t i = 0; i < key_num; i++) {
     EXPECT_FALSE(host_found[i]) << "Key at index " << i << " should not be found after clear";
   }
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试7：随机 keys 测试
 TEST_F(FindAndUpdateTest, random_keys) {
   constexpr size_t dim = 16;
   constexpr size_t key_num = 2048;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_random_keys<K, S, V>(dim, host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), key_num);
 
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   copy_to_host(host_found.get(), device_found, key_num);
   vector<V*> real_values_ptr(key_num, nullptr);
   copy_to_host(real_values_ptr.data(), device_values_ptr, key_num);
 
   size_t found_num = 0;
   for (size_t i = 0; i < key_num; i++) {
     if (host_found[i]) {
       EXPECT_NE(real_values_ptr[i], nullptr);
       found_num++;
 
       vector<V> real_values(dim, 0);
       copy_to_host(real_values.data(), real_values_ptr[i], dim);
       vector<V> expect_values(host_values.begin() + i * dim,
                               host_values.begin() + i * dim + dim);
       EXPECT_EQ(expect_values, real_values);
     }
   }
   EXPECT_EQ(found_num, key_num);
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试8：大规模数据测试
 TEST_F(FindAndUpdateTest, large_scale) {
   constexpr size_t dim = 32;
   constexpr size_t key_num = 64UL * 1024;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), key_num);
 
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   copy_to_host(host_found.get(), device_found, key_num);
 
   size_t found_num = 0;
   for (size_t i = 0; i < key_num; i++) {
     if (host_found[i]) {
       found_num++;
     }
   }
   EXPECT_EQ(found_num, key_num);
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试9：不同 dim 测试 - dim=4
 TEST_F(FindAndUpdateTest, small_dim) {
   constexpr size_t dim = 4;
   constexpr size_t key_num = 512;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), key_num);
 
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   copy_to_host(host_found.get(), device_found, key_num);
 
   size_t found_num = 0;
   for (size_t i = 0; i < key_num; i++) {
     if (host_found[i]) found_num++;
   }
   EXPECT_EQ(found_num, key_num);
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试10：不同 dim 测试 - dim=128
 TEST_F(FindAndUpdateTest, large_dim) {
   constexpr size_t dim = 128;
   constexpr size_t key_num = 256;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
   EXPECT_EQ(table.size(), key_num);
 
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   copy_to_host(host_found.get(), device_found, key_num);
   vector<V*> real_values_ptr(key_num, nullptr);
   copy_to_host(real_values_ptr.data(), device_values_ptr, key_num);
 
   size_t found_num = 0;
   for (size_t i = 0; i < key_num; i++) {
     if (host_found[i]) {
       EXPECT_NE(real_values_ptr[i], nullptr);
       found_num++;
 
       vector<V> real_values(dim, 0);
       copy_to_host(real_values.data(), real_values_ptr[i], dim);
       vector<V> expect_values(host_values.begin() + i * dim,
                               host_values.begin() + i * dim + dim);
       EXPECT_EQ(expect_values, real_values);
     }
   }
   EXPECT_EQ(found_num, key_num);
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试11：通过返回的指针修改值并验证
 TEST_F(FindAndUpdateTest, modify_values_via_ptr) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 128;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 1.0f);  // 初始值为 1.0
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   // 第一次查找获取指针
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   vector<V*> real_values_ptr(key_num, nullptr);
   copy_to_host(real_values_ptr.data(), device_values_ptr, key_num);
 
   // 通过指针修改第一个 key 的值
   vector<V> new_values(dim, 99.0f);
   if (real_values_ptr[0] != nullptr) {
     ASSERT_EQ(aclrtMemcpy(real_values_ptr[0], dim * sizeof(V), new_values.data(),
                           dim * sizeof(V), ACL_MEMCPY_HOST_TO_DEVICE),
               ACL_ERROR_NONE);
   }
 
   // 再次查找验证修改生效
   table.find_and_update(1, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   V* updated_ptr = nullptr;
   copy_to_host(&updated_ptr, device_values_ptr, 1);
   ASSERT_NE(updated_ptr, nullptr);
 
   vector<V> read_values(dim, 0);
   copy_to_host(read_values.data(), updated_ptr, dim);
   EXPECT_EQ(new_values, read_values);
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试12：多次查找同一批 keys
 TEST_F(FindAndUpdateTest, multiple_lookups) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 256;
   constexpr size_t lookup_times = 5;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   // 多次查找同一批 keys
   for (size_t t = 0; t < lookup_times; t++) {
     table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                           nullptr, stream_, true);
     ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
     auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
     copy_to_host(host_found.get(), device_found, key_num);
 
     size_t found_num = 0;
     for (size_t i = 0; i < key_num; i++) {
       if (host_found[i]) found_num++;
     }
     EXPECT_EQ(found_num, key_num) << "Failed at lookup iteration " << t;
   }
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试13：插入后更新值，再查找验证
 TEST_F(FindAndUpdateTest, update_then_find) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 256;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 1.0f);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, nullptr, key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   // 第一次插入
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   // 更新值为 2.0
   vector<V> updated_values(key_num * dim, 2.0f);
   copy_to_device(device_values, updated_values.data(), key_num * dim);
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   // 查找验证值已更新
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   copy_to_host(host_found.get(), device_found, key_num);
   vector<V*> real_values_ptr(key_num, nullptr);
   copy_to_host(real_values_ptr.data(), device_values_ptr, key_num);
 
   for (size_t i = 0; i < key_num; i++) {
     EXPECT_TRUE(host_found[i]);
     if (host_found[i] && real_values_ptr[i] != nullptr) {
       vector<V> real_values(dim, 0);
       copy_to_host(real_values.data(), real_values_ptr[i], dim);
       for (size_t j = 0; j < dim; j++) {
         EXPECT_FLOAT_EQ(real_values[j], 2.0f);
       }
     }
   }
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 
 // 测试14：乱序查询 keys
 TEST_F(FindAndUpdateTest, shuffled_keys_query) {
   constexpr size_t dim = 8;
   constexpr size_t key_num = 512;
 
   HashTableOptions options{
       .init_capacity = init_capacity,
       .max_capacity = init_capacity,
       .max_hbm_for_vectors = hbm_for_values,
       .dim = dim,
       .io_by_cpu = false,
   };
   HashTable<K, V> table;
   table.init(options);
 
   vector<K> host_keys(key_num, 0);
   vector<V> host_values(key_num * dim, 0);
   create_continuous_keys<K, S, V, dim>(host_keys.data(), nullptr, host_values.data(), key_num);
 
   K* device_keys = alloc_device_mem<K>(key_num);
   V* device_values = alloc_device_mem<V>(key_num * dim);
   V** device_values_ptr = alloc_device_mem<V*>(key_num);
   bool* device_found = alloc_device_mem<bool>(key_num);
 
   copy_to_device(device_keys, host_keys.data(), key_num);
   copy_to_device(device_values, host_values.data(), key_num * dim);
 
   table.insert_or_assign(key_num, device_keys, device_values, nullptr, stream_);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   // 打乱 keys 顺序进行查询
   vector<K> shuffled_keys = host_keys;
   std::random_device rd;
   std::mt19937 g(rd());
   std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), g);
 
   copy_to_device(device_keys, shuffled_keys.data(), key_num);
 
   table.find_and_update(key_num, device_keys, device_values_ptr, device_found,
                         nullptr, stream_, true);
   ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_ERROR_NONE);
 
   auto host_found = std::unique_ptr<bool[]>(new bool[key_num]());
   copy_to_host(host_found.get(), device_found, key_num);
 
   size_t found_num = 0;
   for (size_t i = 0; i < key_num; i++) {
     if (host_found[i]) found_num++;
   }
   EXPECT_EQ(found_num, key_num);
 
   free_device_mem(device_keys);
   free_device_mem(device_values);
   free_device_mem(device_values_ptr);
   free_device_mem(device_found);
 }
 