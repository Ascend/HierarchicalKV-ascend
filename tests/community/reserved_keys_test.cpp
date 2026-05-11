/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <iostream>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace community_test_util;

namespace npu {
namespace hkv {

using namespace AscendC;

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_SIZE) inline void
reserved_keys_test_kernel_vf(__gm__ uint64_t* keys, __gm__ bool* results,
                             size_t num_keys, uint32_t blockIdx,
                             uint32_t blockNums) {
  const size_t tid = blockIdx * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < num_keys; i += blockDim.x * blockNums) {
    results[i] = IS_RESERVED_KEY<uint64_t>(keys[i]);
  }
}

__global__ __vector__ void reserved_keys_test_kernel(__gm__ uint64_t* keys,
                                                     __gm__ bool* results,
                                                     size_t num_keys) {
  asc_vf_call<reserved_keys_test_kernel_vf>(
      dim3{static_cast<uint32_t>(BLOCK_SIZE)}, keys, results, num_keys,
      GetBlockIdx(), GetBlockNum());
}

__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_SIZE) inline void
memset64_test_kernel_vf(__gm__ uint64_t* data, uint64_t value,
                        size_t num_elements, uint32_t blockIdx,
                        uint32_t blockNums) {
  const size_t tid = blockIdx * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < num_elements; i += blockDim.x * blockNums) {
    data[i] = value;
  }
}

__global__ __vector__ void memset64_test_kernel(__gm__ uint64_t* data,
                                                uint64_t value,
                                                size_t num_elements) {
  asc_vf_call<memset64_test_kernel_vf>(
      dim3{static_cast<uint32_t>(BLOCK_SIZE)}, data, value, num_elements,
      GetBlockIdx(), GetBlockNum());
}

}  // namespace hkv
}  // namespace npu

namespace {

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

struct ReservedKeySet {
  uint64_t empty_key;
  uint64_t reclaim_key;
  uint64_t locked_key;
};

ReservedKeySet make_reserved_keys(int index) {
  if (index < 1 || index > MAX_RESERVED_KEY_BIT) {
    return {EMPTY_KEY_CPU, RECLAIM_KEY, LOCKED_KEY};
  }

  const uint64_t reserved_key_mask1 = ~(UINT64_C(3) << index);
  const uint64_t reserved_key_mask2 = reserved_key_mask1 & ~UINT64_C(1);
  const uint64_t vacant_key_mask1 = ~(UINT64_C(1) << index);
  const uint64_t vacant_key_mask2 = vacant_key_mask1 & ~UINT64_C(1);

  const uint64_t empty_key = reserved_key_mask2 | (UINT64_C(3) << index);
  const uint64_t reclaim_key = vacant_key_mask2;
  const uint64_t locked_key = empty_key & ~(UINT64_C(2) << index);
  return {empty_key, reclaim_key, locked_key};
}

void memset64_async(uint64_t* dev_ptr, uint64_t value, size_t num_elements,
                   aclrtStream stream) {
  if (num_elements == 0) {
    return;
  }
  constexpr size_t block_size = 1024;
  const size_t grid_size = (num_elements + block_size - 1) / block_size;
  npu::hkv::memset64_test_kernel<<<grid_size, 0, stream>>>(
      dev_ptr, value, num_elements);
}

void test_custom_memset_async() {
  init_env();
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  size_t num_elements = 4;
  uint64_t value = 0xFFFFFFFFFFFFFFF1;
  HostAndDeviceBuffer<uint64_t> dev_ptr;
  dev_ptr.alloc(num_elements, stream);

  memset64_async(dev_ptr.d_data, value, num_elements, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  dev_ptr.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  for (size_t i = 0; i < num_elements; i++) {
    ASSERT_EQ(dev_ptr.h_data[i], value);
  }

  std::cout << "All values were set correctly!" << std::endl;

  dev_ptr.free(stream);
  ACL_CHECK(aclrtDestroyStream(stream));
}

void test_reserved_keys(const uint64_t* test_keys, const bool* expected_results,
                      size_t num_keys) {
  init_env();
  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  HostAndDeviceBuffer<uint64_t> d_keys;
  HostAndDeviceBuffer<bool> d_results;
  d_keys.alloc(num_keys, stream);
  d_results.alloc(num_keys, stream);
  d_keys.set_from_host(test_keys, num_keys, stream);

  constexpr size_t block_size = 256;
  const size_t num_blocks = (num_keys + block_size - 1) / block_size;

  npu::hkv::reserved_keys_test_kernel<<<num_blocks, 0, stream>>>(
      d_keys.d_data, d_results.d_data, num_keys);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  d_results.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < num_keys; i++) {
    ASSERT_EQ(d_results.h_data[i], expected_results[i]);
  }

  d_results.free(stream);
  d_keys.free(stream);
  ACL_CHECK(aclrtDestroyStream(stream));
  std::cout << "All tests passed." << std::endl;
}

void test_key_options() {
  for (int i = 0; i <= MAX_RESERVED_KEY_BIT; i++) {
    ACL_CHECK(init_reserved_keys(i));
    const ReservedKeySet reserved_keys = make_reserved_keys(i);

    uint64_t test_keys[6] = {reserved_keys.empty_key, reserved_keys.reclaim_key,
                            reserved_keys.locked_key, UINT64_C(0x0),
                            UINT64_C(0x10), EMPTY_KEY_CPU};
    bool expected_results[6] = {true,  true,  true,
                               false, false, (i == 0) ? true : false};
    test_reserved_keys(test_keys, expected_results, 4);
  }
}

}  // namespace

TEST(ReservedKeysTest, test_key_options) {
  test_key_options();
  test_custom_memset_async();
}
