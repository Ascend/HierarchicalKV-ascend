/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "aclrtlaunch_read_from_ptr_kernel.h"

#define UNEQUAL_EXPR(expr1, expr2)                             \
  {                                                            \
    std::cout << __FILE__ << ":" << __LINE__ << ":Unequal\n"   \
              << "\t\t" << #expr1 << " != " << #expr2 << "\n"; \
  }

#define HKV_EXPECT_TRUE(cond, msg)                                       \
  if ((cond) == false) {                                                 \
    fprintf(stderr, "[ERROR] %s at %s : %d\n", msg, __FILE__, __LINE__); \
    exit(-1);                                                            \
  }

namespace test_util {

template <class K, class S>
void create_random_keys(K* h_keys, S* h_scores, int KEY_NUM,
                        int freq_range = 1000) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    h_scores[i] = num % freq_range;
    i++;
  }
}

template <class K, class S, class V, size_t DIM = 16>
void create_random_keys(K* h_keys, S* h_scores, V* h_vectors, size_t KEY_NUM,
                        size_t range = std::numeric_limits<uint64_t>::max()) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng) % range);
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_scores != nullptr) {
      h_scores[i] = num;
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < DIM; j++) {
        h_vectors[i * DIM + j] = static_cast<float>(num * 0.00001);
      }
    }
    i++;
  }
}

template <class K>
void create_random_bools(bool* bools, int KEY_NUM, float true_ratio = 0.6) {
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;

  for (int i = 0; i < KEY_NUM; i++) {
    K bound = 1000 * true_ratio;
    bools[i] = (distr(eng) % 1000 < bound);
  }
}

template <class K, class S, class V>
void create_random_keys(size_t dim, K* h_keys, S* h_scores, V* h_vectors,
                        int KEY_NUM,
                        size_t range = std::numeric_limits<uint64_t>::max()) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng) % range);
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_scores != nullptr) {
      h_scores[i] = num;
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < dim; j++) {
        h_vectors[i * dim + j] = static_cast<V>(num * 0.00001);
      }
    }
    i++;
  }
}

template <class K, class S, class V>
void create_random_keys_advanced(
    size_t dim, K* h_keys, S* h_scores, V* h_vectors, int KEY_NUM,
    size_t range = std::numeric_limits<uint64_t>::max(), int freq_range = 10) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng) % range);
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_scores != nullptr) {
      h_scores[i] = num % freq_range;
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < dim; j++) {
        h_vectors[i * dim + j] = static_cast<float>(num * 0.00001);
      }
    }
    i++;
  }
}

template <class K, class S, class V>
void create_random_keys_advanced(
    size_t dim, K* h_keys, K* pre_h_keys, S* h_scores, V* h_vectors,
    int KEY_NUM, size_t range = std::numeric_limits<uint64_t>::max(),
    int freq_range = 10, float repeat_rate = 0.9) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  std::mt19937_64 eng_switch(rd());
  std::uniform_int_distribution<K> distr_switch;
  int i = 0;
  int pre_pos = 0;

  while (numbers.size() < KEY_NUM) {
    bool repeated = static_cast<K>(distr_switch(eng_switch) % 100000) <
                    static_cast<K>(repeat_rate * 100000);
    if (repeated) {
      numbers.insert(pre_h_keys[pre_pos++]);
    } else {
      numbers.insert(distr(eng) % range);
    }
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_scores != nullptr) {
      h_scores[i] = num % freq_range;
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < dim; j++) {
        h_vectors[i * dim + j] = static_cast<float>(num * 0.00001);
      }
    }
    i++;
  }
}

inline uint64_t Murmur3HashHost(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

template <class K, class S, class V, size_t DIM = 16>
void create_continuous_keys(K* h_keys, S* h_scores, V* h_vectors, int KEY_NUM,
                            K start = 1) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
    if (h_scores != nullptr) {
      h_scores[i] = h_keys[i];
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < DIM; j++) {
        h_vectors[i * DIM + j] = static_cast<V>(h_keys[i] * 0.00001);
      }
    }
  }
}

template <class K, class S, class V, size_t DIM = 16>
void create_keys_in_one_buckets(K* h_keys, S* h_scores, V* h_vectors,
                                int KEY_NUM, int capacity,
                                int bucket_max_size = 128, int bucket_idx = 0,
                                K min = 0,
                                K max = static_cast<K>(0xFFFFFFFFFFFFFFFD)) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  K candidate;
  K hashed_key;
  size_t global_idx;
  size_t bkt_idx;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    candidate = (distr(eng) % (max - min)) + min;
    hashed_key = Murmur3HashHost(candidate);
    global_idx = hashed_key & (capacity - 1);
    bkt_idx = global_idx / bucket_max_size;
    if (bkt_idx == bucket_idx) {
      numbers.insert(candidate);
    }
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_scores != nullptr) {
      h_scores[i] = num;
    }
    for (size_t j = 0; j < DIM; j++) {
      *(h_vectors + i * DIM + j) = static_cast<float>(num * 0.00001);
    }
    i++;
  }
}

template <class K, class S, class V, size_t DIM = 16>
void create_keys_in_one_buckets_lfu(K* h_keys, S* h_scores, V* h_vectors,
                                    int KEY_NUM, int capacity,
                                    int bucket_max_size = 128,
                                    int bucket_idx = 0, K min = 0,
                                    K max = static_cast<K>(0xFFFFFFFFFFFFFFFD),
                                    int freq_range = 1000) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  K candidate;
  K hashed_key;
  size_t global_idx;
  size_t bkt_idx;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    candidate = (distr(eng) % (max - min)) + min;
    hashed_key = Murmur3HashHost(candidate);
    global_idx = hashed_key & (capacity - 1);
    bkt_idx = global_idx / bucket_max_size;
    if (bkt_idx == bucket_idx) {
      numbers.insert(candidate);
    }
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_scores != nullptr) {
      h_scores[i] = num % freq_range;
    }
    for (size_t j = 0; j < DIM; j++) {
      *(h_vectors + i * DIM + j) = static_cast<float>(num * 0.00001);
    }
    i++;
  }
}

template <class S>
S make_expected_score_for_epochlfu(S global_epoch, S original_score) {
  bool if_overflow = (original_score >= static_cast<S>(0xFFFFFFFF));
  return ((global_epoch << 32) | (if_overflow ? (static_cast<S>(0xFFFFFFFF))
                                              : original_score & 0xFFFFFFFF));
}

template <class V>
void read_from_ptr(V** __restrict src, V* __restrict dst, const size_t dim,
                   size_t n, aclrtStream stream) {
  const size_t block_size = 1024;
  const size_t N = n * dim;
  const size_t grid_size = (N - 1) / block_size + 1;
  HKV_EXPECT_TRUE((grid_size <= 65535), "Pointer is already assigned.");

  ACLRT_LAUNCH_KERNEL(read_from_ptr_kernel)
  (grid_size, stream, src, dst, dim, N, sizeof(V));
}

inline void init_env() {
  static bool init_flag = false;
  if (!init_flag) {
    HKV_EXPECT_TRUE((aclInit(nullptr) == ACL_ERROR_NONE), "aclInit failed!");
    auto device_id_env = std::getenv("HKV_TEST_DEVICE");
    int32_t device_id = device_id_env != nullptr ? std::stoi(device_id_env) : 0;
    HKV_EXPECT_TRUE((aclrtSetDevice(device_id) == ACL_ERROR_NONE),
                    "aclrtSetDevice failed");
    init_flag = true;
  }
}

inline uint32_t round_up8(const uint32_t x) {
  constexpr uint32_t round_size = 8;
  if (x % round_size != 0) {
    return (x / round_size + 1) * round_size;
  }
  return x;
}
}  // namespace test_util
