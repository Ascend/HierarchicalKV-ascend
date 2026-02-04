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

#include <gtest/gtest.h>
#include <vector>
#include <functional>
#include "acl/acl.h"

#include "aclrtlaunch_test_desired_when_missed_kernel.h"
#include "aclrtlaunch_test_update_kernel.h"
#include "aclrtlaunch_test_update_with_digest_kernel.h"
#include "aclrtlaunch_test_update_without_missed_bucket_kernel.h"
#include "aclrtlaunch_test_update_without_missed_ptr_kernel.h"

#include "../test_util.h"
#include "../../include/types.h"

using namespace std;
using namespace npu::hkv;
using namespace test_util;

using K = uint64_t;
using V = float;
using S = uint64_t;
using D = uint8_t;

constexpr uint32_t CAP = 128;

struct DeviceMem {
  DeviceMem(size_t size) : size_(size) { aclrtMalloc(&ptr_, size, ACL_MEM_MALLOC_HUGE_FIRST); }
  ~DeviceMem() { if (ptr_) aclrtFree(ptr_); }
  void* get() const { return ptr_; }
  void zero() { aclrtMemset(ptr_, size_, 0, size_); }
  template<typename T> void copy_from(const vector<T>& data) {
    aclrtMemcpy(ptr_, data.size() * sizeof(T), data.data(), 
                data.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
  }
  template<typename T> void copy_to(vector<T>& data) const {
    aclrtMemcpy(data.data(), data.size() * sizeof(T), ptr_,
                data.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  }
private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

class ScoreFunctorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    init_env();
    aclrtCreateStream(&stream_);
  }
  void TearDown() override { if (stream_) aclrtDestroyStream(stream_); }

  S run_single(function<void(void*)> launcher) {
    DeviceMem out(sizeof(S));
    launcher(out.get());
    aclrtSynchronizeStream(stream_);
    vector<S> out_data = {0};
    out.copy_to(out_data);
    return out_data[0];
  }

  // 创建设备上的 Bucket 结构体
  Bucket<K, V, S>* alloc_bucket() {
    // 1. 分配 Bucket 结构体
    Bucket<K, V, S>* d_bucket = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&d_bucket), sizeof(Bucket<K, V, S>), 
                ACL_MEM_MALLOC_HUGE_FIRST);

    // 2. 分配 keys/scores/digests 内存（布局：digests → keys → scores）
    uint32_t reserve = CAP < 128 ? 128 : CAP;
    void* d_mem = nullptr;
    aclrtMalloc(&d_mem, reserve * sizeof(D) + CAP * (sizeof(K) + sizeof(S)),
                ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(d_mem, reserve * sizeof(D) + CAP * (sizeof(K) + sizeof(S)), 0,
                reserve * sizeof(D) + CAP * (sizeof(K) + sizeof(S)));

    // 3. 初始化 Bucket 的指针成员
    Bucket<K, V, S> h_bucket;
    h_bucket.digests_ = reinterpret_cast<D*>(d_mem);
    h_bucket.keys_ = reinterpret_cast<K*>(h_bucket.digests_ + reserve);
    h_bucket.scores_ = reinterpret_cast<S*>(h_bucket.keys_ + CAP);
    h_bucket.vectors = nullptr;

    // 4. 拷贝到设备
    aclrtMemcpy(d_bucket, sizeof(Bucket<K, V, S>), &h_bucket, sizeof(Bucket<K, V, S>),
                ACL_MEMCPY_HOST_TO_DEVICE);

    return d_bucket;
  }

  // 从 Bucket 获取 keys 地址（用于 K* bucket_keys 参数）
  K* get_keys_from_bucket(Bucket<K, V, S>* d_bucket) {
    Bucket<K, V, S> h_bucket;
    aclrtMemcpy(&h_bucket, sizeof(Bucket<K, V, S>), d_bucket, sizeof(Bucket<K, V, S>),
                ACL_MEMCPY_DEVICE_TO_HOST);
    return h_bucket.keys_;
  }

  // 释放 Bucket 结构体
  void free_bucket(Bucket<K, V, S>* d_bucket) {
    if (!d_bucket) return;

    // 读取 Bucket 结构体
    Bucket<K, V, S> h_bucket;
    aclrtMemcpy(&h_bucket, sizeof(Bucket<K, V, S>), d_bucket, sizeof(Bucket<K, V, S>),
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    // 释放内存块
    if (h_bucket.digests_) aclrtFree(h_bucket.digests_);
    
    // 释放 Bucket 结构体
    aclrtFree(d_bucket);
  }
  
  // 读取 scores
  vector<S> read_scores(Bucket<K, V, S>* d_bucket) {
    Bucket<K, V, S> h_bucket;
    aclrtMemcpy(&h_bucket, sizeof(Bucket<K, V, S>), d_bucket, sizeof(Bucket<K, V, S>),
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    vector<S> scores(CAP);
    aclrtMemcpy(scores.data(), CAP * sizeof(S), h_bucket.scores_, CAP * sizeof(S),
                ACL_MEMCPY_DEVICE_TO_HOST);
    return scores;
  }
  
  // 写入 scores
  void write_scores(Bucket<K, V, S>* d_bucket, const vector<S>& scores) {
    Bucket<K, V, S> h_bucket;
    aclrtMemcpy(&h_bucket, sizeof(Bucket<K, V, S>), d_bucket, sizeof(Bucket<K, V, S>),
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    aclrtMemcpy(h_bucket.scores_, CAP * sizeof(S), scores.data(), CAP * sizeof(S),
                ACL_MEMCPY_HOST_TO_DEVICE);
  }
  
  // 读取 digests
  vector<D> read_digests(Bucket<K, V, S>* d_bucket) {
    Bucket<K, V, S> h_bucket;
    aclrtMemcpy(&h_bucket, sizeof(Bucket<K, V, S>), d_bucket, sizeof(Bucket<K, V, S>),
                ACL_MEMCPY_DEVICE_TO_HOST);
    
    vector<D> result(CAP);
    aclrtMemcpy(result.data(), CAP * sizeof(D), h_bucket.digests_, CAP * sizeof(D),
                ACL_MEMCPY_DEVICE_TO_HOST);
    return result;
  }

  aclrtStream stream_ = nullptr;
};

// desired_when_missed
TEST_F(ScoreFunctorTest, desired_lru) {
  S cycle = 9876543210UL;
  S result = run_single([&](void* out) {
    ACLRT_LAUNCH_KERNEL(test_desired_when_missed_kernel)(
        1, stream_, EvictStrategyInternal::kLru, 0, 0, 0, cycle, out);
  });
  EXPECT_EQ(result, cycle);
}

TEST_F(ScoreFunctorTest, desired_lfu) {
    S result = run_single([&](void* out) {
      ACLRT_LAUNCH_KERNEL(test_desired_when_missed_kernel)(
          1, stream_, EvictStrategyInternal::kLfu, 0, 0, 0, 0, out);
    });
    EXPECT_EQ(result, MAX_SCORE);
  }
  
  TEST_F(ScoreFunctorTest, desired_epoch_lru) {
    S epoch = 5;
    S cycle = 0x100000000UL;
    S result = run_single([&](void* out) {
      ACLRT_LAUNCH_KERNEL(test_desired_when_missed_kernel)(
          1, stream_, EvictStrategyInternal::kEpochLru, 0, 0, epoch, cycle, out);
    });
    EXPECT_EQ(result, (epoch << 32) | ((cycle >> 20) & 0xFFFFFFFFUL));
  }
  
  TEST_F(ScoreFunctorTest, desired_epoch_lfu) {
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{100});
    S epoch = 7;
    S result = run_single([&](void* out) {
      ACLRT_LAUNCH_KERNEL(test_desired_when_missed_kernel)(
          1, stream_, EvictStrategyInternal::kEpochLfu, in.get(), 0, epoch, 0, out);
    });
    EXPECT_EQ(result, (epoch << 32) | 100);
  }
  
  TEST_F(ScoreFunctorTest, desired_customized) {
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{12345});
    S result = run_single([&](void* out) {
      ACLRT_LAUNCH_KERNEL(test_desired_when_missed_kernel)(
          1, stream_, EvictStrategyInternal::kCustomized, in.get(), 0, 0, 0, out);
    });
    EXPECT_EQ(result, 12345);
  }
  
  // update (使用 BUCKET* 参数)
  TEST_F(ScoreFunctorTest, update_lru) {
    auto bucket = alloc_bucket();
    S desired = 123456;
    
    ACLRT_LAUNCH_KERNEL(test_update_kernel)(
        1, stream_, EvictStrategyInternal::kLru, bucket, 0, 0, 0, desired, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_lfu) {
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{999});
    auto bucket = alloc_bucket();
    
    // 新插入
    ACLRT_LAUNCH_KERNEL(test_update_kernel)(
        1, stream_, EvictStrategyInternal::kLfu, bucket, 0, in.get(), 0, 111, true);
    aclrtSynchronizeStream(stream_);
    EXPECT_EQ(read_scores(bucket)[0], 999);
    
    // 累加
    ACLRT_LAUNCH_KERNEL(test_update_kernel)(
        1, stream_, EvictStrategyInternal::kLfu, bucket, 0, in.get(), 0, MAX_SCORE - 1, false);
    aclrtSynchronizeStream(stream_);
    EXPECT_EQ(read_scores(bucket)[0], 999 + 999);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_epoch_lru) {
    auto bucket = alloc_bucket();
    S desired = (5UL << 32) | 123;
    
    ACLRT_LAUNCH_KERNEL(test_update_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLru, bucket, 0, 0, 0, desired, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_epoch_lfu) {
    auto bucket = alloc_bucket();
    S desired = (7UL << 32) | 200;
    
    ACLRT_LAUNCH_KERNEL(test_update_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLfu, bucket, 0, 0, 0, desired, true);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
  
    ACLRT_LAUNCH_KERNEL(test_update_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLfu, bucket, 0, 0, 0, desired, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], (7UL << 32) | 400);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_customized) {
    auto bucket = alloc_bucket();
    S desired = 88888;
    
    ACLRT_LAUNCH_KERNEL(test_update_kernel)(
        1, stream_, EvictStrategyInternal::kCustomized, bucket, 0, 0, 0, desired, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
    free_bucket(bucket);
  }
  
  // update_with_digest (使用 K* bucket_keys 参数)
  TEST_F(ScoreFunctorTest, update_with_digest_lru) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    S desired = 123;
    D digest = 0x55;
    
    ACLRT_LAUNCH_KERNEL(test_update_with_digest_kernel)(
        1, stream_, EvictStrategyInternal::kLru, keys, 0, 0, 0, desired, CAP, digest, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
    EXPECT_EQ(read_digests(bucket)[0], digest);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_with_digest_lfu) {
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{333});
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    D digest = 0xAA;
    
    ACLRT_LAUNCH_KERNEL(test_update_with_digest_kernel)(
        1, stream_, EvictStrategyInternal::kLfu, keys, 0, in.get(), 0, 0, CAP, digest, true);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], 333);
    EXPECT_EQ(read_digests(bucket)[0], digest);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_with_digest_epoch_lru) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    S desired = (3UL << 32) | 456;
    D digest = 0xBB;
    
    ACLRT_LAUNCH_KERNEL(test_update_with_digest_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLru, keys, 0, 0, 0, desired, CAP, digest, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
    EXPECT_EQ(read_digests(bucket)[0], digest);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_with_digest_epoch_lfu) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    S desired = (9UL << 32) | 777;
    D digest = 0xCC;
    
    ACLRT_LAUNCH_KERNEL(test_update_with_digest_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLfu, keys, 0, 0, 0, desired, CAP, digest, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
    EXPECT_EQ(read_digests(bucket)[0], digest);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_with_digest_customized) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    S desired = 99999;
    D digest = 0xDD;
    
    ACLRT_LAUNCH_KERNEL(test_update_with_digest_kernel)(
        1, stream_, EvictStrategyInternal::kCustomized, keys, 0, 0, 0, desired, CAP, digest, false);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], desired);
    EXPECT_EQ(read_digests(bucket)[0], digest);
    free_bucket(bucket);
  }
  
  // update_without_missed - bucket 版本 (使用 BUCKET* 参数)
  TEST_F(ScoreFunctorTest, update_without_missed_bucket_lru) {
    auto bucket = alloc_bucket();
    S cycle = 555555;
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_bucket_kernel)(
        1, stream_, EvictStrategyInternal::kLru, bucket, 0, 0, 0, 0, cycle);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], cycle);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_bucket_lfu) {
    auto bucket = alloc_bucket();
    write_scores(bucket, vector<S>(CAP, 10));
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{20});
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_bucket_kernel)(
        1, stream_, EvictStrategyInternal::kLfu, bucket, 0, in.get(), 0, 0, 0);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], 10 + 20);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_bucket_epoch_lru) {
    auto bucket = alloc_bucket();
    S epoch = 11;
    S cycle = 0x200000000UL;
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_bucket_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLru, bucket, 0, 0, 0, epoch, cycle);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], (epoch << 32) | ((cycle >> 20) & 0xFFFFFFFFUL));
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_bucket_epoch_lfu) {
    auto bucket = alloc_bucket();
    S epoch = 13;
    write_scores(bucket, vector<S>(CAP, (epoch << 32) | 50));
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{20});
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_bucket_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLfu, bucket, 0, in.get(), 0, epoch, 0);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0] & 0xFFFFFFFFUL, 70);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_bucket_customized) {
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{77777});
    auto bucket = alloc_bucket();
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_bucket_kernel)(
        1, stream_, EvictStrategyInternal::kCustomized, bucket, 0, in.get(), 0, 0, 0);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], 77777);
    free_bucket(bucket);
  }
  
  // update_without_missed - ptr 版本 (使用 K* bucket_keys 参数)
  TEST_F(ScoreFunctorTest, update_without_missed_ptr_lru) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    S cycle = 666666;
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_ptr_kernel)(
        1, stream_, EvictStrategyInternal::kLru, keys, CAP, 0, 0, 0, 0, cycle);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], cycle);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_ptr_lfu) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    write_scores(bucket, vector<S>(CAP, 20));
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{20});
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_ptr_kernel)(
        1, stream_, EvictStrategyInternal::kLfu, keys, CAP, 0, in.get(), 0, 0, 0);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], 20 + 20);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_ptr_epoch_lru) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    S epoch = 15;
    S cycle = 0x300000000UL;
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_ptr_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLru, keys, CAP, 0, 0, 0, epoch, cycle);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], (epoch << 32) | ((cycle >> 20) & 0xFFFFFFFFUL));
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_ptr_epoch_lfu) {
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    S epoch = 17;
    write_scores(bucket, vector<S>(CAP, (epoch << 32) | 60));
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{20});
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_ptr_kernel)(
        1, stream_, EvictStrategyInternal::kEpochLfu, keys, CAP, 0, in.get(), 0, epoch, 0);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0] & 0xFFFFFFFFUL, 80);
    free_bucket(bucket);
  }
  
  TEST_F(ScoreFunctorTest, update_without_missed_ptr_customized) {
    DeviceMem in(sizeof(S));
    in.copy_from(vector<S>{88888});
    auto bucket = alloc_bucket();
    auto keys = get_keys_from_bucket(bucket);
    
    ACLRT_LAUNCH_KERNEL(test_update_without_missed_ptr_kernel)(
        1, stream_, EvictStrategyInternal::kCustomized, keys, CAP, 0, in.get(), 0, 0, 0);
    aclrtSynchronizeStream(stream_);
    
    EXPECT_EQ(read_scores(bucket)[0], 88888);
    free_bucket(bucket);
  }