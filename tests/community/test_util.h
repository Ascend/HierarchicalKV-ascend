/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <assert.h>
#include <gtest/gtest.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "hkv_hashtable.h"
#include "kernels/utils_kernel/utils_kernel.h"

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

namespace npu {
namespace hkv {

inline uint32_t get_test_value_move_block_dim() {
  auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  HKV_EXPECT_TRUE((platform != nullptr), "get platform failed");
  const uint32_t block_dim = platform->GetCoreNumAiv();
  HKV_EXPECT_TRUE((block_dim != 0), "get vector core num failed");
  return block_dim;
}

}  // namespace hkv
}  // namespace npu

namespace test_util {

constexpr size_t DEFAULT_DIM = 8;
constexpr size_t DEFAULT_CAPACITY = 128UL * 1024;

inline bool& simd_value_move_enabled() {
  static bool enabled = false;
  return enabled;
}

inline void set_simd_value_move_enabled(bool enabled) {
  simd_value_move_enabled() = enabled;
}

inline bool is_simd_value_move_enabled() {
  return simd_value_move_enabled();
}

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

template <class K, class S, class V>
void create_continuous_keys(size_t dim, K* h_keys, S* h_scores, V* h_vectors,
                            int KEY_NUM, K start = 1) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
    if (h_scores != nullptr) {
      h_scores[i] = h_keys[i];
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < dim; j++) {
        h_vectors[i * dim + j] = static_cast<V>(h_keys[i] * 0.00001);
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

template <typename T, size_t DIM>
struct ValueArray {
 public:
  T data[DIM];

  T sum() const {
    T s = 0;
    for (size_t i = 0; i < DIM; ++i) {
      s += data[i];
    }
    return s;
  }

  T operator[](size_t i) const { return data[i]; }
};

template <typename T>
struct HostAndDeviceBuffer {
 public:
  void alloc(size_t n, aclrtStream stream = 0) {
    clear_storage(stream);
    HKV_EXPECT_TRUE((aclrtMalloc(reinterpret_cast<void**>(&d_data),
                                 n * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) ==
                     ACL_ERROR_NONE),
                    "aclrtMalloc failed");
    h_data = static_cast<T*>(std::malloc(n * sizeof(T)));
    HKV_EXPECT_TRUE((h_data != nullptr), "malloc failed");
    size_ = n;
    to_zeros(stream);
  }

  ~HostAndDeviceBuffer() { clear_storage(); }

  void free(aclrtStream stream = 0) {
    clear_storage(stream);
  }

  void clear_storage(aclrtStream stream = 0) {
    if (d_data != nullptr) {
      if (stream != nullptr) {
        HKV_EXPECT_TRUE((aclrtSynchronizeStream(stream) == ACL_ERROR_NONE),
                        "aclrtSynchronizeStream failed");
      }
      HKV_EXPECT_TRUE((aclrtFree(d_data) == ACL_ERROR_NONE), "aclrtFree failed");
      d_data = nullptr;
    }
    if (h_data != nullptr) {
      std::free(h_data);
      h_data = nullptr;
    }
    size_ = 0;
  }

  void set_from_host(const T* data, size_t n, aclrtStream stream = 0) {
    HKV_EXPECT_TRUE((n <= size_), "SetFromHost size exceeds allocation");
    std::memcpy(h_data, data, n * sizeof(T));
    sync_data(true, stream);
  }

  void set_from_device(const T* data, size_t n, aclrtStream stream = 0) {
    HKV_EXPECT_TRUE((n <= size_), "SetFromDevice size exceeds allocation");
    HKV_EXPECT_TRUE(
        (aclrtMemcpyAsync(d_data, n * sizeof(T), data, n * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_DEVICE, stream) ==
         ACL_ERROR_NONE),
        "aclrtMemcpyAsync device to device failed");
    HKV_EXPECT_TRUE(
        (aclrtMemcpyAsync(h_data, n * sizeof(T), data, n * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_HOST, stream) ==
         ACL_ERROR_NONE),
        "aclrtMemcpyAsync device to host failed");
  }

  bool set_value_in_range(T start, T skip, size_t stripe,
                       aclrtStream stream = 0) {
    if (h_data == nullptr || skip == 0 || stripe == 0 || size_ % stripe != 0) {
      return false;
    }

    const size_t stripe_num = size_ / stripe;
    for (size_t i = 0; i < stripe_num; ++i) {
      const T value = start + static_cast<T>(i) * skip;
      for (size_t j = 0; j < stripe; ++j) {
        h_data[i * stripe + j] = value;
      }
    }
    sync_data(true, stream);
    return true;
  }

  void to_zeros(aclrtStream stream = 0) {
    if (d_data != nullptr) {
      HKV_EXPECT_TRUE((aclrtMemset(d_data, size_ * sizeof(T), 0,
                                   size_ * sizeof(T)) == ACL_ERROR_NONE),
                      "aclrtMemset failed");
    }
    if (h_data != nullptr) {
      std::memset(h_data, 0, size_ * sizeof(T));
    }
    if (stream != nullptr) {
      HKV_EXPECT_TRUE((aclrtSynchronizeStream(stream) == ACL_ERROR_NONE),
                      "aclrtSynchronizeStream failed");
    }
  }

  void to_const(const T val, aclrtStream stream = 0) {
    for (size_t i = 0; i < size_; ++i) {
      h_data[i] = val;
    }
    sync_data(true, stream);
  }

  void sync_data(bool h2d, aclrtStream stream = 0) {
    if (size_ == 0 || h_data == nullptr || d_data == nullptr) {
      return;
    }
    if (h2d) {
      HKV_EXPECT_TRUE(
          (aclrtMemcpyAsync(d_data, size_ * sizeof(T), h_data, size_ * sizeof(T),
                            ACL_MEMCPY_HOST_TO_DEVICE, stream) ==
           ACL_ERROR_NONE),
          "aclrtMemcpyAsync host to device failed");
    } else {
      HKV_EXPECT_TRUE(
          (aclrtMemcpyAsync(h_data, size_ * sizeof(T), d_data, size_ * sizeof(T),
                            ACL_MEMCPY_DEVICE_TO_HOST, stream) ==
           ACL_ERROR_NONE),
          "aclrtMemcpyAsync device to host failed");
    }
  }

 public:
  T* h_data = nullptr;
  T* d_data = nullptr;
  size_t size_ = 0;
};

template <typename K, typename V, typename S>
struct KVMSBuffer {
 public:
  KVMSBuffer() : len_(0), dim_(0) {}

  void reserve(size_t n, size_t dim, aclrtStream stream = 0) {
    keys.alloc(n, stream);
    values.alloc(n * dim, stream);
    scores.alloc(n, stream);
    status.alloc(n, stream);
    len_ = n;
    dim_ = dim;
  }

  ~KVMSBuffer() { free(); }

  void free(aclrtStream stream = 0) {
    keys.free(stream);
    values.free(stream);
    scores.free(stream);
    status.free(stream);
    len_ = 0;
    dim_ = 0;
  }

  size_t len() const { return len_; }
  size_t dim() const { return dim_; }

  void to_range(size_t start, size_t skip = 1, aclrtStream stream = 0) {
    HKV_EXPECT_TRUE(
        keys.set_value_in_range(static_cast<K>(start), static_cast<K>(skip), 1,
                             stream),
        "keys SetValueInRange failed");
    HKV_EXPECT_TRUE(values.set_value_in_range(static_cast<V>(start),
                                           static_cast<V>(skip), dim_, stream),
                    "values SetValueInRange failed");
    status.to_zeros(stream);
  }

  void to_zeros(aclrtStream stream = 0) {
    keys.to_zeros(stream);
    values.to_zeros(stream);
    scores.to_zeros(stream);
    status.to_zeros(stream);
  }

  void set_score(const S score, aclrtStream stream = 0) {
    scores.to_const(score, stream);
  }

  K* keys_ptr(bool on_device = true) {
    return on_device ? keys.d_data : keys.h_data;
  }

  V* values_ptr(bool on_device = true) {
    return on_device ? values.d_data : values.h_data;
  }

  S* scores_ptr(bool on_device = true) {
    return on_device ? scores.d_data : scores.h_data;
  }

  bool* status_ptr(bool on_device = true) {
    return on_device ? status.d_data : status.h_data;
  }

  void sync_data(bool h2d, aclrtStream stream = 0) {
    keys.sync_data(h2d, stream);
    values.sync_data(h2d, stream);
    scores.sync_data(h2d, stream);
    status.sync_data(h2d, stream);
  }

  void copy_from(KVMSBuffer<K, V, S>& src, aclrtStream stream = 0) {
    HKV_EXPECT_TRUE((src.len() == len_), "CopyFrom len mismatch");
    HKV_EXPECT_TRUE((src.dim() == dim_), "CopyFrom dim mismatch");
    std::memcpy(keys_ptr(false), src.keys_ptr(false), sizeof(K) * len());
    std::memcpy(scores_ptr(false), src.scores_ptr(false), sizeof(S) * len());
    std::memcpy(values_ptr(false), src.values_ptr(false),
                sizeof(V) * len() * dim());
    keys.sync_data(true, stream);
    values.sync_data(true, stream);
    scores.sync_data(true, stream);
    status.sync_data(true, stream);
  }

 public:
  HostAndDeviceBuffer<K> keys;
  HostAndDeviceBuffer<V> values;
  HostAndDeviceBuffer<S> scores;
  HostAndDeviceBuffer<bool> status;
  size_t len_;
  size_t dim_;
};

template <class V>
void read_from_ptr(V** __restrict src, V* __restrict dst, const size_t dim,
                   size_t n, aclrtStream stream) {
  if (is_simd_value_move_enabled()) {
    const uint32_t block_dim = npu::hkv::get_test_value_move_block_dim();
    HKV_EXPECT_TRUE(
        (n <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())),
        "read_from_ptr n exceeds uint32_t");
    auto tiling =
        npu::hkv::GetValueMoveTiling(static_cast<uint32_t>(n), block_dim,
                                     static_cast<uint32_t>(dim), sizeof(V),
                                     false);
    npu::hkv::read_value_kernel<V>
        <<<block_dim, tiling.valid_ub_size, stream>>>(
            tiling.former_num, tiling.former_core_move_num,
            tiling.tail_core_move_num, tiling.tile_size, tiling.num_tiles,
            static_cast<uint32_t>(dim), dst, n, src);
    return;
  }

  const size_t block_size = 1024;
  const size_t N = n * dim;
  const size_t grid_size = (N - 1) / block_size + 1;
  HKV_EXPECT_TRUE((grid_size <= 65535), "Pointer is already assigned.");
  npu::hkv::read_from_ptr_kernel<V>
      <<<grid_size, 0, stream>>>(reinterpret_cast<void*>(src), dst, dim, N);
}

}  // namespace test_util

namespace npu {
namespace hkv {

using namespace AscendC;

template <class V>
__global__ __vector__ void read_or_write_value_test_kernel(
    uint32_t former_num, uint64_t former_core_move_num,
    uint64_t tail_core_move_num, uint32_t tile_size, uint32_t num_tiles,
    uint32_t dim, __gm__ V* param_values, uint64_t n,
    __gm__ V* __gm__* table_value_addrs, __gm__ bool* founds) {
  uint64_t cur_block_idx = GetBlockIdx();
  uint64_t core_start_idx = 0;
  uint64_t core_move_count = 0;
  constexpr uint32_t BUFFER_NUM = 2;
  if (cur_block_idx < former_num) {
    core_start_idx = cur_block_idx * former_core_move_num;
    core_move_count = former_core_move_num;
  } else {
    core_start_idx = former_num * former_core_move_num +
                     (cur_block_idx - former_num) * tail_core_move_num;
    core_move_count = tail_core_move_num;
  }

  TPipe pipe;
  TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> move_queue;
  pipe.InitBuffer(move_queue, BUFFER_NUM, tile_size * sizeof(V));

  GlobalTensor<V> table_values_addr_gm;
  GlobalTensor<V> param_values_gm;
  LocalTensor<V> move_local;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};

  param_values_gm.SetGlobalBuffer(param_values);

  for (uint64_t i = core_start_idx; i < core_start_idx + core_move_count; i++) {
    __gm__ V* table_value_addr = table_value_addrs[i];
    if (table_value_addr == nullptr) {
      continue;
    }

    uint64_t param_offset = i * dim;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      uint32_t current_tile_size = (tile_idx == num_tiles - 1)
                                       ? (dim - tile_idx * tile_size)
                                       : tile_size;
      DataCopyExtParams copy_params{
          1, static_cast<uint32_t>(current_tile_size * sizeof(V)), 0, 0, 0};

      move_queue.AllocTensor<V>(move_local);
      table_values_addr_gm.SetGlobalBuffer(table_value_addr +
                                           tile_idx * tile_size);

      if (founds[i]) {
        DataCopyPad(move_local, table_values_addr_gm, copy_params, pad_params);
        move_queue.EnQue<V>(move_local);
        move_queue.DeQue<V>(move_local);
        DataCopyPad(param_values_gm[param_offset + tile_idx * tile_size],
                    move_local, copy_params);
      } else {
        DataCopyPad(move_local,
                    param_values_gm[param_offset + tile_idx * tile_size],
                    copy_params, pad_params);
        move_queue.EnQue<V>(move_local);
        move_queue.DeQue<V>(move_local);
        DataCopyPad(table_values_addr_gm, move_local, copy_params);
      }

      move_queue.FreeTensor(move_local);
    }
  }
}

template <class V>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_SIZE) inline void
read_or_write_ptr_kernel_vf(__gm__ void* src_addr, __gm__ V* dst_addr,
                            __gm__ bool* read_or_write, const size_t dim,
                            size_t N, uint32_t blockIdx,
                            uint32_t blockNums) {
  __gm__ V* const __gm__* src =
      reinterpret_cast<__gm__ V* const __gm__* __restrict>(src_addr);
  __gm__ V* dst = reinterpret_cast<__gm__ V* __restrict>(dst_addr);

  const size_t tid = (blockIdx * blockDim.x) + threadIdx.x;
  for (size_t t = tid; t < N; t += blockDim.x * blockNums) {
    const size_t vec_index = t / dim;
    const size_t dim_index = t % dim;
    if (src[vec_index] == nullptr) {
      continue;
    }
    if (read_or_write[vec_index]) {
      dst[t] = src[vec_index][dim_index];
    } else {
      src[vec_index][dim_index] = dst[t];
    }
  }
}

template <class V>
__global__ __vector__ void read_or_write_ptr_kernel(__gm__ void* src,
                                                    __gm__ V* dst,
                                                    __gm__ bool* read_or_write,
                                                    const size_t dim,
                                                    size_t N) {
  asc_vf_call<read_or_write_ptr_kernel_vf<V>>(
      dim3{static_cast<uint32_t>(BLOCK_SIZE)}, src, dst, read_or_write, dim, N,
      GetBlockIdx(), GetBlockNum());
}

}  // namespace hkv
}  // namespace npu

namespace test_util {

template <class V>
void read_or_write_ptr(V** __restrict src, V* __restrict dst,
                       bool* __restrict read_or_write, const size_t dim,
                       size_t n, aclrtStream stream) {
  if (is_simd_value_move_enabled()) {
    const uint32_t block_dim = npu::hkv::get_test_value_move_block_dim();
    HKV_EXPECT_TRUE(
        (n <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())),
        "read_or_write_ptr n exceeds uint32_t");
    auto tiling =
        npu::hkv::GetValueMoveTiling(static_cast<uint32_t>(n), block_dim,
                                     static_cast<uint32_t>(dim), sizeof(V),
                                     false);
    npu::hkv::read_or_write_value_test_kernel<V>
        <<<block_dim, tiling.valid_ub_size, stream>>>(
            tiling.former_num, tiling.former_core_move_num,
            tiling.tail_core_move_num, tiling.tile_size, tiling.num_tiles,
            static_cast<uint32_t>(dim), dst, n, src, read_or_write);
    return;
  }

  const size_t block_size = 1024;
  const size_t N = n * dim;
  const size_t grid_size = (N - 1) / block_size + 1;
  HKV_EXPECT_TRUE((grid_size <= 65535), "Pointer is already assigned.");
  npu::hkv::read_or_write_ptr_kernel<V><<<grid_size, 0, stream>>>(
      reinterpret_cast<void*>(src), dst, read_or_write, dim, N);
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

template <typename K, typename V, typename S,
          int Strategy = npu::hkv::EvictStrategy::kLru>
std::unique_ptr<npu::hkv::HashTable<K, V, S, Strategy>> get_table(
    size_t dim, size_t capacity, size_t num_of_buckets_per_alloc) {
  using Table = npu::hkv::HashTable<K, V, S, Strategy>;
  auto table = std::make_unique<Table>();

  size_t total_mem = 0;
  size_t free_mem = 0;
  constexpr size_t hbm_for_values = 4UL << 30;
  EXPECT_EQ(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem),
            ACL_ERROR_NONE);
  EXPECT_GT(free_mem, hbm_for_values)
      << "free HBM is not enough free:" << free_mem
      << "need:" << hbm_for_values;

  npu::hkv::HashTableOptions options{
      .init_capacity = capacity,
      .max_capacity = capacity,
      .max_hbm_for_vectors = hbm_for_values,
      .dim = dim,
      .io_by_cpu = false,
      .num_of_buckets_per_alloc = num_of_buckets_per_alloc,
  };

  table->init(options);

  return table;
}

template <typename K, typename V, typename S>
std::unique_ptr<npu::hkv::HashTable<K, V, S>> get_default_table() {
  return get_table<K, V, S>(DEFAULT_DIM, DEFAULT_CAPACITY, 1);
}

// 辅助函数：使用 key 初始化 values
template <class K, class V>
void init_value_using_key(K* h_keys, V* h_vectors, const size_t key_num,
                          size_t dim) {
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < dim; j++) {
      h_vectors[i * dim + j] = static_cast<V>(h_keys[i] * 0.00001);
    }
  }
}

// 辅助函数：验证两个 bool 数组是否完全相等
inline bool all_equal_npu(const bool* d_array1, const bool* d_array2,
                        size_t size, aclrtStream stream) {
  std::vector<char> h_array1(size);
  std::vector<char> h_array2(size);

  HKV_EXPECT_TRUE(
      (aclrtMemcpy(h_array1.data(), size * sizeof(char), const_cast<bool*>(d_array1),
                   size * sizeof(char), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE),
      "aclrtMemcpy device to host failed");

  HKV_EXPECT_TRUE(
      (aclrtMemcpy(h_array2.data(), size * sizeof(char), const_cast<bool*>(d_array2),
                   size * sizeof(char), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE),
      "aclrtMemcpy device to host failed");

  HKV_EXPECT_TRUE((aclrtSynchronizeStream(stream) == ACL_ERROR_NONE),
                  "aclrtSynchronizeStream failed");

  for (size_t i = 0; i < size; i++) {
    if (h_array1[i] != h_array2[i]) {
      std::cout << "Mismatch at index " << i << ": " << (int)h_array1[i]
                << " != " << (int)h_array2[i] << std::endl;
      return false;
    }
  }
  return true;
}

// 辅助函数：验证 bool 数组是否全为 true
inline bool all_true_npu(const bool* d_array, size_t size, aclrtStream stream) {
  std::vector<char> h_array(size);

  HKV_EXPECT_TRUE(
      (aclrtMemcpy(h_array.data(), size * sizeof(char), const_cast<bool*>(d_array),
                   size * sizeof(char), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_ERROR_NONE),
      "aclrtMemcpy device to host failed");

  HKV_EXPECT_TRUE((aclrtSynchronizeStream(stream) == ACL_ERROR_NONE),
                  "aclrtSynchronizeStream failed");

  for (size_t i = 0; i < size; i++) {
    if (!h_array[i]) {
      std::cout << "False found at index " << i << std::endl;
      return false;
    }
  }
  return true;
}
}  // namespace test_util

namespace community_test_util = test_util;
