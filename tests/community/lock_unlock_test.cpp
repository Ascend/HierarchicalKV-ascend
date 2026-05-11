/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace community_test_util;

namespace {

constexpr size_t dim = 64;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using TableOptions = HashTableOptions;

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

void test_lock_and_unlock() {
  init_env();
  TableOptions opt;

  // table setting
  const size_t U = 4 * 1024 * 1024UL;
  const size_t M = 65536UL;
  opt.max_capacity = U;
  opt.init_capacity = U;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.num_of_buckets_per_alloc = 8;

  using Table = HashTable<i64, f32, u64, EvictStrategy::kCustomized>;
  opt.dim = dim;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  HostAndDeviceBuffer<bool> d_found;
  HostAndDeviceBuffer<bool> d_lock_results;
  HostAndDeviceBuffer<i64*> lock_keys_ptr;
  d_found.alloc(M, stream);
  d_lock_results.alloc(M, stream);
  lock_keys_ptr.alloc(M, stream);

  // step1
  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  // step2
  KVMSBuffer<i64, f32, u64> buffer;
  buffer.reserve(M, dim, stream);

  i64 start = 0;
  for (int i = 0; i < static_cast<int>(U / M); i++) {
    buffer.to_range(start, 1, stream);
    start += static_cast<i64>(M);
    buffer.set_score(static_cast<u64>(i), stream);
    table->insert_or_assign(M, buffer.keys_ptr(), buffer.values_ptr(),
                            buffer.scores_ptr(), stream);

    d_found.to_zeros(stream);
    d_lock_results.to_zeros(stream);
    table->contains(M, buffer.keys_ptr(), d_found.d_data, stream);
    table->lock_keys(M, buffer.keys_ptr(), lock_keys_ptr.d_data,
                     d_lock_results.d_data, stream, buffer.scores_ptr());
    ACL_CHECK(aclrtSynchronizeStream(stream));
    bool result =
        all_equal_npu(d_found.d_data, d_lock_results.d_data, M, stream);
    ASSERT_EQ(result, true);
    result = all_true_npu(d_found.d_data, M, stream);
    ASSERT_EQ(result, true);

    d_found.to_zeros(stream);
    d_lock_results.to_zeros(stream);
    table->contains(M, buffer.keys_ptr(), d_found.d_data, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    result =
        all_equal_npu(d_found.d_data, d_lock_results.d_data, M, stream);
    ASSERT_EQ(result, true);

    d_found.to_zeros(stream);
    table->unlock_keys(M, lock_keys_ptr.d_data, buffer.keys_ptr(),
                       d_lock_results.d_data, stream);
    table->contains(M, buffer.keys_ptr(), d_found.d_data, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    result =
        all_equal_npu(d_found.d_data, d_lock_results.d_data, M, stream);
    ASSERT_EQ(result, true);
    result = all_true_npu(d_found.d_data, M, stream);
    ASSERT_EQ(result, true);
  }

  buffer.free(stream);
  lock_keys_ptr.free(stream);
  d_lock_results.free(stream);
  d_found.free(stream);
  ACL_CHECK(aclrtDestroyStream(stream));
}

}  // namespace

TEST(LockAndUnlockTest, test_lock_and_unlock) { test_lock_and_unlock(); }
