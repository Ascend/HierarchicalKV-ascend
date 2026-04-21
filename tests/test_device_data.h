/*
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

#pragma once

#include <vector>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

template <typename K, typename V, typename S>
class DeviceData {
 public:
  DeviceData() = default;
  DeviceData(const DeviceData&) = delete;
  DeviceData& operator=(const DeviceData&) = delete;

  ~DeviceData() {
    if (device_keys != nullptr) {
      EXPECT_EQ(aclrtFree(device_keys), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_values), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_values_ptr), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_found), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtFree(device_scores), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_ERROR_NONE);
      EXPECT_EQ(aclrtDestroyStream(stream), ACL_ERROR_NONE);
    }
  }

  void malloc(size_t key_num, size_t dim = test_util::DEFAULT_DIM) {
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_keys),
                          key_num * each_key_size, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&device_values),
                    key_num * each_value_size * dim, ACL_MEM_MALLOC_HUGE_FIRST),
        ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_values_ptr),
                          key_num * sizeof(V*), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_found),
                          key_num * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&device_scores),
                          key_num * each_score_size, ACL_MEM_MALLOC_HUGE_FIRST),
              ACL_ERROR_NONE);
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_ERROR_NONE);
  }

  void copy_keys(std::vector<K>& host_keys, size_t key_num) {
    ASSERT_EQ(
        aclrtMemcpy(device_keys, key_num * each_key_size, host_keys.data(),
                    key_num * each_key_size, ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_ERROR_NONE);
  }

  void copy_values(std::vector<V>& host_values, size_t key_num,
                   size_t dim = test_util::DEFAULT_DIM) {
    ASSERT_EQ(aclrtMemcpy(device_values, key_num * each_value_size * dim,
                          host_values.data(), key_num * each_value_size * dim,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  void copy_scores(std::vector<S>& host_scores, size_t key_num) {
    ASSERT_EQ(aclrtMemcpy(device_scores, key_num * each_score_size,
                          host_scores.data(), key_num * each_score_size,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_ERROR_NONE);
  }

  aclrtStream stream = nullptr;
  K* device_keys = nullptr;
  V* device_values = nullptr;
  V** device_values_ptr = nullptr;
  bool* device_found = nullptr;
  S* device_scores = nullptr;
  size_t each_key_size = sizeof(K);
  size_t each_score_size = sizeof(S);
  size_t each_value_size = sizeof(V);
};

template <typename K, typename V, typename S>
void check_result(std::vector<V>& host_values, size_t key_num,
                  DeviceData<K, V, S>& device_data,
                  size_t expect_found_num = std::numeric_limits<size_t>::max(),
                  size_t dim = test_util::DEFAULT_DIM) {
  expect_found_num = expect_found_num == std::numeric_limits<size_t>::max()
                         ? key_num
                         : expect_found_num;
  bool* host_found = nullptr;
  ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&host_found),
                            key_num * sizeof(bool)),
            ACL_ERROR_NONE);
  ASSERT_EQ(
      aclrtMemcpy(host_found, key_num * sizeof(bool), device_data.device_found,
                  key_num * sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST),
      ACL_ERROR_NONE);
  std::vector<void*> real_values_ptr(key_num, nullptr);
  ASSERT_EQ(aclrtMemcpy(real_values_ptr.data(), key_num * sizeof(void*),
                        device_data.device_values_ptr, key_num * sizeof(void*),
                        ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_ERROR_NONE);
  size_t found_num = 0;
  std::vector<V> real_values(dim, 0);
  for (size_t i = 0; i < key_num; i++) {
    if (host_found[i]) {
      ASSERT_NE(real_values_ptr[i], nullptr);
      found_num++;

      ASSERT_EQ(
          aclrtMemcpy(real_values.data(), dim * device_data.each_value_size,
                      real_values_ptr[i], dim * device_data.each_value_size,
                      ACL_MEMCPY_DEVICE_TO_HOST),
          ACL_ERROR_NONE);
      std::vector<V> expect_values(host_values.begin() + i * dim,
                                   host_values.begin() + i * dim + dim);
      ASSERT_EQ(expect_values, real_values);
    } else {
      ASSERT_EQ(real_values_ptr[i], nullptr);
    }
  }
  EXPECT_EQ(found_num, expect_found_num);

  ASSERT_EQ(aclrtFreeHost(host_found), ACL_ERROR_NONE);
}