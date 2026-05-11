/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
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
#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "test_util.h"

using namespace npu::hkv;
using namespace community_test_util;

namespace {

using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using TableOptions = HashTableOptions;
using Table = HashTable<i64, f32, u64, EvictStrategy::kCustomized>;

#define ACL_CHECK(expr) ASSERT_EQ((expr), ACL_ERROR_NONE)

template <class K, class S>
struct ExportIfPredFunctor {
  __forceinline__ __simt_callee__ bool operator()(const K& key, const S& score,
                                                  const K& pattern,
                                                  const S& threshold) {
    (void)key;
    (void)pattern;
    return score < threshold;
  }
};

template <class K, class V, class S>
struct ExportIfPredFunctorV2 {
  K pattern;
  S threshold;
  ExportIfPredFunctorV2(K pattern, S threshold)
      : pattern(pattern), threshold(threshold) {}

  template <int GroupSize>
  __forceinline__ __simt_callee__ bool operator()(const K& key,
                                                  const __gm__ V* value,
                                                  const S& score) {
    (void)value;
    (void)GroupSize;
    return ((!npu::hkv::IS_RESERVED_KEY<K>(key)) && (score < threshold));
  }
};

template <class K, class V, class S>
struct ExportIfPredFunctorV3 {
  K pattern;
  S threshold;
  uint32_t dim;

  ExportIfPredFunctorV3(K pattern, S threshold)
      : pattern(pattern), threshold(threshold), dim(0) {}

  template <int GroupSize>
  __forceinline__ __simt_callee__ bool operator()(const K& key,
                                                  const __gm__ V* value,
                                                  const S& score) {
    (void)GroupSize;
    bool pred = ((!npu::hkv::IS_RESERVED_KEY<K>(key)) && (score < threshold));
    for (uint32_t i = 0; pred && i < dim; ++i) {
      if (value[i] != static_cast<V>(key)) {
        pred = false;
      }
    }
    return pred;
  }
};

// Using for_each API to simulate export_batch_if_v2 API.
template <class K, class V, class S>
struct ForEachExecutionFuncV4 {
  K pattern;
  S threshold;
  uint32_t dim;
  __gm__ size_t* d_counter;
  __gm__ K* out_keys;
  __gm__ V* out_vals;
  __gm__ S* out_scores;

  ForEachExecutionFuncV4(K pattern, S threshold)
      : pattern(pattern),
        threshold(threshold),
        dim(0),
        d_counter(nullptr),
        out_keys(nullptr),
        out_vals(nullptr),
        out_scores(nullptr) {}

  __forceinline__ __simt_callee__ void operator()(const K& key,
                                                  __gm__ V* value,
                                                  __gm__ S* score,
                                                  int32_t group_size) {
    const S score_val = *score;
    const bool match =
        (!npu::hkv::IS_RESERVED_KEY<K>(key)) && (score_val < threshold);
    const uint32_t vote = asc_ballot(match);
    const uint32_t group_cnt = AscendC::Simt::Popc(vote);
    const int32_t lane = threadIdx.x % group_size;
    size_t group_offset = 0;
    if (lane == 0) {
      group_offset = atomicAdd(d_counter, static_cast<size_t>(group_cnt));
    }
    group_offset = asc_shfl(group_offset, 0, group_size);

    const uint32_t previous_cnt =
        group_cnt - AscendC::Simt::Popc(vote >> lane);
    if (match) {
      const size_t out_pos = group_offset + previous_cnt;
      out_keys[out_pos] = key;
      if (out_scores != nullptr) {
        out_scores[out_pos] = score_val;
      }
      for (uint32_t i = 0; i < dim; ++i) {
        out_vals[out_pos * dim + i] = value[i];
      }
    }
  }
};

enum class ExportIfVersion { V1, V2, V3, V4 };

struct DeviceCounter {
  size_t* ptr = nullptr;

  ~DeviceCounter() {
    if (ptr != nullptr) {
      aclrtFree(ptr);
    }
  }

  void alloc() {
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptr), sizeof(size_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  }

  void reset(aclrtStream stream) {
    ACL_CHECK(aclrtMemsetAsync(ptr, sizeof(size_t), 0, sizeof(size_t), stream));
  }

  void to_host(size_t* host, aclrtStream stream) {
    ACL_CHECK(aclrtMemcpyAsync(host, sizeof(size_t), ptr, sizeof(size_t),
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }
};

template <ExportIfVersion EV>
void test_export_batch_if_with_limited_size() {
  constexpr uint64_t CAP = 1llu << 24;
  const size_t n0 = (1llu << 23) - 163;
  const size_t n1 = (1llu << 23) + 221;
  const size_t n2 = (1llu << 23) - 17;
  constexpr size_t dim = 64;
  i64 pattern = 0;
  u64 threshold = 40;

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  TableOptions options;
  options.init_capacity = CAP;
  options.max_capacity = CAP;
  options.dim = dim;
  options.max_hbm_for_vectors = GB(100);

  auto table = std::make_unique<Table>();
  table->init(options);

  DeviceCounter d_cnt;
  d_cnt.alloc();

  KVMSBuffer<i64, f32, u64> buffer0;
  buffer0.reserve(n0, dim, stream);
  buffer0.to_range(0, 1, stream);
  buffer0.set_score(static_cast<u64>(15), stream);
  {
    KVMSBuffer<i64, f32, u64> buffer0_ev;
    buffer0_ev.reserve(n0, dim, stream);
    buffer0_ev.to_zeros(stream);
    d_cnt.reset(stream);
    table->insert_and_evict(n0, buffer0.keys_ptr(), buffer0.values_ptr(),
                            buffer0.scores_ptr(), buffer0_ev.keys_ptr(),
                            buffer0_ev.values_ptr(), buffer0_ev.scores_ptr(),
                            d_cnt.ptr, stream, true, false);
    const size_t table_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ASSERT_EQ(table_size, n0);
  }

  KVMSBuffer<i64, f32, u64> buffer1;
  buffer1.reserve(n1, dim, stream);
  buffer1.to_range(n0, 1, stream);
  buffer1.set_score(static_cast<u64>(30), stream);
  {
    KVMSBuffer<i64, f32, u64> buffer1_ev;
    buffer1_ev.reserve(n0, dim, stream);
    buffer1_ev.to_zeros(stream);
    d_cnt.reset(stream);
    table->insert_and_evict(n0, buffer1.keys_ptr(), buffer1.values_ptr(),
                            buffer1.scores_ptr(), buffer1_ev.keys_ptr(),
                            buffer1_ev.values_ptr(), buffer1_ev.scores_ptr(),
                            d_cnt.ptr, stream, true, false);
    (void)table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
  }

  KVMSBuffer<i64, f32, u64> buffer2;
  buffer2.reserve(n2, dim, stream);
  buffer2.to_range(n0 + n1, 1, stream);
  buffer2.set_score(static_cast<u64>(45), stream);
  {
    KVMSBuffer<i64, f32, u64> buffer2_ev;
    buffer2_ev.reserve(n0, dim, stream);
    buffer2_ev.to_zeros(stream);
    d_cnt.reset(stream);
    table->insert_and_evict(n0, buffer2.keys_ptr(), buffer2.values_ptr(),
                            buffer2.scores_ptr(), buffer2_ev.keys_ptr(),
                            buffer2_ev.values_ptr(), buffer2_ev.scores_ptr(),
                            d_cnt.ptr, stream, true, false);
    const size_t table_size = table->size(stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::printf("final size: %zu, capacity: %zu\n", table_size,
                table->capacity());
  }

  table->size_if<ExportIfPredFunctor>(pattern, threshold, d_cnt.ptr, stream);
  size_t h_cnt = 0;
  d_cnt.to_host(&h_cnt, stream);
  std::printf("---> check h_cnt from size_if kernel: %zu\n", h_cnt);

  KVMSBuffer<i64, f32, u64> buffer_out;
  buffer_out.reserve(h_cnt, dim, stream);
  buffer_out.to_zeros(stream);
  d_cnt.reset(stream);

  if (EV == ExportIfVersion::V1) {
    table->export_batch_if<ExportIfPredFunctor>(
        pattern, threshold, static_cast<size_t>(CAP), 0, d_cnt.ptr,
        buffer_out.keys_ptr(), buffer_out.values_ptr(),
        buffer_out.scores_ptr(), stream);
  } else if (EV == ExportIfVersion::V2) {
    ExportIfPredFunctorV2<i64, f32, u64> pred(pattern, threshold);
    table->export_batch_if_v2<ExportIfPredFunctorV2<i64, f32, u64>>(
        pred, static_cast<size_t>(CAP), 0, d_cnt.ptr, buffer_out.keys_ptr(),
        buffer_out.values_ptr(), buffer_out.scores_ptr(), stream);
  } else if (EV == ExportIfVersion::V3) {
    ExportIfPredFunctorV3<i64, f32, u64> pred(pattern, threshold);
    pred.dim = dim;
    table->export_batch_if_v2<ExportIfPredFunctorV3<i64, f32, u64>>(
        pred, static_cast<size_t>(CAP), 0, d_cnt.ptr, buffer_out.keys_ptr(),
        buffer_out.values_ptr(), buffer_out.scores_ptr(), stream);
  } else if (EV == ExportIfVersion::V4) {
    ForEachExecutionFuncV4<i64, f32, u64> f(pattern, threshold);
    f.dim = dim;
    f.d_counter = d_cnt.ptr;
    f.out_keys = buffer_out.keys_ptr();
    f.out_vals = buffer_out.values_ptr();
    f.out_scores = buffer_out.scores_ptr();
    table->for_each(0, static_cast<size_t>(CAP), f, stream);
  }

  size_t h_cnt2 = 0;
  d_cnt.to_host(&h_cnt2, stream);
  std::printf("final h_cnt2: %zu\n", h_cnt2);
  ASSERT_EQ(h_cnt, h_cnt2)
      << "size_if and export_batch_if get different matching count.";

  buffer_out.sync_data(false, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::unordered_map<i64, u64> record;
  for (size_t i = 0; i < h_cnt; ++i) {
    const i64 key = buffer_out.keys_ptr(false)[i];
    const u64 score = buffer_out.scores_ptr(false)[i];
    ASSERT_LT(score, threshold);
    record[key] = score;
    for (size_t j = 0; j < dim; ++j) {
      const f32 value = buffer_out.values_ptr(false)[i * dim + j];
      ASSERT_EQ(key, static_cast<i64>(value));
    }
  }
  ASSERT_EQ(record.size(), h_cnt2);
  std::printf("record: %zu\n", record.size());
  std::printf("n0+n1: %zu\n", n0 + n1);
  std::printf("n0+n1+n2: %zu\n", n0 + n1 + n2);
  std::printf("done\n");

  ACL_CHECK(aclrtDestroyStream(stream));
}

}  // namespace

TEST(ExportBatchIfTest, test_export_batch_if_with_limited_size) {
  init_env();
  test_export_batch_if_with_limited_size<ExportIfVersion::V1>();
  test_export_batch_if_with_limited_size<ExportIfVersion::V2>();
  test_export_batch_if_with_limited_size<ExportIfVersion::V3>();
  test_export_batch_if_with_limited_size<ExportIfVersion::V4>();
}
