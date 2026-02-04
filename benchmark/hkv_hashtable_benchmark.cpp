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

#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "acl/acl.h"
#include "hkv_hashtable.h"
#include "benchmark_util.h"
#include "debug.h"
#include "tiling/platform/platform_ascendc.h"

using std::cerr;
using std::cout;
using std::endl;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

using namespace npu::hkv;
using namespace benchmark;

enum class Test_Mode {
  pure_hbm = 0,
  hybrid = 1,
};

const float EPSILON = 0.001f;

std::string rep(int n) { return std::string(n, ' '); }

using K = uint64_t;
using S = uint64_t;
using V = float;
using EvictStrategy = npu::hkv::EvictStrategy;
using TableOptions = npu::hkv::HashTableOptions;

static uint32_t g_core_num_aiv = 0;

constexpr uint32_t FLOAT_TYPR_BYTES = 4;
constexpr int64_t RANDOM_MOD = 0;
constexpr int64_t CONST_MOD = 1;
constexpr int64_t VALUE_OFFSET = sizeof(int64_t) * 3;  // key + counter + flags

inline uint32_t ROUND_UP8(const uint32_t x) {
  constexpr uint32_t ROUND_SIZE = 8;
  if (x % ROUND_SIZE != 0) {
    return (x / ROUND_SIZE + 1) * ROUND_SIZE;
  }
  return x;
}

template <class Table>
float test_one_api(std::shared_ptr<Table>& table, const API_Select api,
                   const size_t dim, const size_t init_capacity,
                   const size_t key_num_per_op, const float load_factor,
                   const float hitrate = 0.6f) {
  K* h_keys;
  S* h_scores;
  V* h_vectors;
  bool* h_found;

  NPU_CHECK(aclrtMallocHost((void**)&h_keys, key_num_per_op * sizeof(K)));
  NPU_CHECK(aclrtMallocHost((void**)&h_scores, key_num_per_op * sizeof(S)));
  NPU_CHECK(
      aclrtMallocHost((void**)&h_vectors, key_num_per_op * sizeof(V) * dim));
  NPU_CHECK(aclrtMallocHost((void**)&h_found, key_num_per_op * sizeof(bool)));

  NPU_CHECK(aclrtMemset(h_vectors, key_num_per_op * sizeof(V) * dim, 0,
                        key_num_per_op * sizeof(V) * dim));

  bool need_scores = (Table::evict_strategy == EvictStrategy::kLfu ||
                      Table::evict_strategy == EvictStrategy::kEpochLfu ||
                      Table::evict_strategy == EvictStrategy::kCustomized);

  K* d_keys;
  S* d_scores_real;
  S* d_scores;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;
  K* d_keys_out;

  K* d_evict_keys;
  S* d_evict_scores;

  NPU_CHECK(aclrtMalloc((void**)&d_keys, key_num_per_op * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_scores_real, key_num_per_op * sizeof(S),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_vectors, key_num_per_op * sizeof(V) * dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_def_val, key_num_per_op * sizeof(V) * dim,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr, key_num_per_op * sizeof(V*),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_found, key_num_per_op * sizeof(bool),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_keys_out, key_num_per_op * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));

  NPU_CHECK(aclrtMalloc((void**)&d_evict_keys, key_num_per_op * sizeof(K),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  NPU_CHECK(aclrtMalloc((void**)&d_evict_scores, key_num_per_op * sizeof(S),
                        ACL_MEM_MALLOC_HUGE_FIRST));

  NPU_CHECK(aclrtMemset(d_vectors, key_num_per_op * sizeof(V) * dim, 1,
                        key_num_per_op * sizeof(V) * dim));
  NPU_CHECK(aclrtMemset(d_def_val, key_num_per_op * sizeof(V) * dim, 2,
                        key_num_per_op * sizeof(V) * dim));
  NPU_CHECK(aclrtMemset(d_vectors_ptr, key_num_per_op * sizeof(V*), 0,
                        key_num_per_op * sizeof(V*)));
  NPU_CHECK(aclrtMemset(d_found, key_num_per_op * sizeof(bool), 0,
                        key_num_per_op * sizeof(bool)));

  d_scores = need_scores ? d_scores_real : nullptr;

  aclrtStream stream;
  NPU_CHECK(aclrtCreateStream(&stream));

  // initialize insert
  // step 1, no need to load load_factor
  uint64_t key_num_init = static_cast<uint64_t>(init_capacity * load_factor);
  const float target_load_factor = key_num_init * 1.0f / init_capacity;
  uint64_t key_num_remain = key_num_init % key_num_per_op == 0
                                ? key_num_per_op
                                : key_num_init % key_num_per_op;
  int32_t loop_num_init = (key_num_init + key_num_per_op - 1) / key_num_per_op;

  K start = 0UL;

  S threshold = host_nano<S>();
  int global_epoch = 0;
  for (; global_epoch < loop_num_init; global_epoch++) {
    table->set_global_epoch(global_epoch);
    uint64_t key_num_cur_insert =
        global_epoch == loop_num_init - 1 ? key_num_remain : key_num_per_op;
    create_continuous_keys<K, S>(h_keys, h_scores, key_num_cur_insert, start);
    NPU_CHECK(aclrtMemcpy(d_keys, key_num_cur_insert * sizeof(K), h_keys,
                          key_num_cur_insert * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_scores_real, key_num_cur_insert * sizeof(S),
                          h_scores, key_num_cur_insert * sizeof(S),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    table->find_or_insert(key_num_cur_insert, d_keys, d_vectors_ptr, d_found,
                          d_scores, stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));

    start += key_num_cur_insert;
  }

  // step 2
  float real_load_factor = table->load_factor(stream);
  NPU_CHECK(aclrtSynchronizeStream(stream));
  while (target_load_factor - real_load_factor > EPSILON) {
    auto key_num_append = static_cast<int64_t>(
        (target_load_factor - real_load_factor) * init_capacity);
    if (key_num_append <= 0) break;
    key_num_append =
        std::min(static_cast<int64_t>(key_num_per_op), key_num_append);
    create_continuous_keys<K, S>(h_keys, h_scores, key_num_append, start);
    NPU_CHECK(aclrtMemcpy(d_keys, key_num_append * sizeof(K), h_keys,
    key_num_append * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_scores_real, key_num_append * sizeof(S),
    h_scores, key_num_append * sizeof(S),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    table->insert_or_assign(key_num_append, d_keys, d_vectors, d_scores,
                            stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    start += key_num_append;
    real_load_factor = table->load_factor(stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));
  }

  // For trigger the kernel selection in advance.
  int key_num_per_op_warmup = 1;
  for (int i = 0; i < 9; i++, global_epoch++) {
    table->set_global_epoch(global_epoch);
    switch (api) {
      case API_Select::find: {
        table->find(key_num_per_op_warmup, d_keys, d_vectors, d_found, d_scores,
                    stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        break;
      }
      case API_Select::insert_or_assign: {
        table->insert_or_assign(key_num_per_op_warmup, d_keys, d_vectors,
                                d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        break;
      }
      case API_Select::find_or_insert: {
        table->find_or_insert(key_num_per_op_warmup, d_keys, d_vectors,
                              d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        break;
      }
      case API_Select::assign: {
        table->assign(key_num_per_op_warmup, d_keys, d_def_val, d_scores,
                      stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        break;
      }
      case API_Select::insert_and_evict: {
        table->insert_and_evict(key_num_per_op_warmup, d_keys, d_vectors,
                                d_scores, d_evict_keys, d_def_val,
                                d_evict_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        break;
      }
      case API_Select::find_ptr: {
        V** d_vectors_ptr = nullptr;
        bool* d_found = nullptr;
        NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr,
                              key_num_per_op_warmup * sizeof(V*),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        NPU_CHECK(aclrtMalloc((void**)&d_found,
                              key_num_per_op_warmup * sizeof(bool),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        // benchmark::array2ptr(d_vectors_ptr, d_vectors, dim,
        //                      key_num_per_op_warmup, stream);

        NPU_CHECK(aclrtSynchronizeStream(stream));
        table->find(key_num_per_op_warmup, d_keys, d_vectors_ptr, d_found, d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        // benchmark::read_from_ptr(d_vectors_ptr, d_vectors, dim,
        //                          key_num_per_op_warmup, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        NPU_CHECK(aclrtFree(d_vectors_ptr));
        NPU_CHECK(aclrtFree(d_found));
        break;
      }
      case API_Select::find_or_insert_ptr: {
        V** d_vectors_ptr = nullptr;
        bool* d_found;
        NPU_CHECK(aclrtMalloc((void**)&d_found,
                              key_num_per_op_warmup * sizeof(bool),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr,
                              key_num_per_op_warmup * sizeof(V*),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        // benchmark::array2ptr(d_vectors_ptr, d_vectors, dim,
        //                      key_num_per_op_warmup, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        table->find_or_insert(key_num_per_op_warmup, d_keys, d_vectors_ptr,
                              d_found, d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        NPU_CHECK(aclrtFree(d_vectors_ptr));
        NPU_CHECK(aclrtFree(d_found));
        break;
      }
      case API_Select::export_batch: {
        size_t* d_dump_counter = nullptr;
        NPU_CHECK(aclrtMalloc((void**)&d_dump_counter, sizeof(size_t),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        NPU_CHECK(
            aclrtMemset(d_dump_counter, sizeof(size_t), 0, sizeof(size_t)));

        table->export_batch(key_num_per_op_warmup, 0, d_dump_counter, d_keys,
                            d_vectors, d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        NPU_CHECK(aclrtFree(d_dump_counter));
        break;
      }
      case API_Select::export_batch_if: {
        size_t* d_dump_counter = nullptr;
        NPU_CHECK(aclrtMalloc((void**)&d_dump_counter, sizeof(size_t),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        NPU_CHECK(
            aclrtMemset(d_dump_counter, sizeof(size_t), 0, sizeof(size_t)));
        K pattern = 0;
        table->template export_batch_if<ExportIfPredFunctor>(
            pattern, threshold, key_num_per_op_warmup, 0, d_dump_counter,
            d_keys, d_vectors, d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        NPU_CHECK(aclrtFree(d_dump_counter));
        break;
      }
      case API_Select::contains: {
        table->contains(1, d_keys, d_found, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        break;
      }
      case API_Select::find_and_update: {
        V** d_vectors_ptr = nullptr;
        bool* d_found = nullptr;
        NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr, key_num_per_op_warmup * sizeof(V*),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        NPU_CHECK(aclrtMalloc((void**)&d_found, key_num_per_op_warmup * sizeof(bool),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        NPU_CHECK(aclrtSynchronizeStream(stream));
        table->find_and_update(key_num_per_op_warmup, d_keys, d_vectors_ptr, d_found, d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        NPU_CHECK(aclrtFree(d_vectors_ptr));
        NPU_CHECK(aclrtFree(d_found));
        break;
      }
      case API_Select::assign_scores: {
        table->assign_scores(key_num_per_op_warmup, d_keys, d_scores, stream);
        NPU_CHECK(aclrtSynchronizeStream(stream));
        break;
      }
      default: {
        std::cout << "[Unsupport API]\n";
      }
    }
  }
  create_keys_for_hitrate<K, S>(h_keys, h_scores, key_num_per_op, hitrate,
                                Hit_Mode::last_insert, start, true /*reset*/);
  NPU_CHECK(aclrtMemcpy(d_keys, key_num_per_op * sizeof(K), h_keys,
                        key_num_per_op * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
  NPU_CHECK(aclrtMemcpy(d_scores_real, key_num_per_op * sizeof(K), h_scores,
                        key_num_per_op * sizeof(K), ACL_MEMCPY_HOST_TO_DEVICE));
  auto timer = benchmark::Timer<double>();
  global_epoch++;
  table->set_global_epoch(global_epoch);
  switch (api) {
    case API_Select::find: {
      timer.start();
      table->find(key_num_per_op, d_keys, d_vectors, d_found, d_scores, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      break;
    }
    case API_Select::insert_or_assign: {
      timer.start();
      table->insert_or_assign(key_num_per_op, d_keys, d_vectors, d_scores,
                              stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      break;
    }
    case API_Select::find_or_insert: {
      timer.start();
      table->find_or_insert(key_num_per_op, d_keys, d_vectors, d_scores,
                            stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      break;
    }
    case API_Select::assign: {
      timer.start();
      table->assign(key_num_per_op, d_keys, d_def_val, d_scores, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      break;
    }
    case API_Select::insert_and_evict: {
      timer.start();
      table->insert_and_evict(key_num_per_op, d_keys, d_vectors, d_scores,
                              d_evict_keys, d_def_val, d_evict_scores, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      break;
    }
    case API_Select::find_ptr: {
      V** d_vectors_ptr = nullptr;
      bool* d_found = nullptr;
      NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr, key_num_per_op * sizeof(V*),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      NPU_CHECK(aclrtMalloc((void**)&d_found, key_num_per_op * sizeof(bool),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      // benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op,
      //                      stream);

      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.start();
      table->find(key_num_per_op, d_keys, d_vectors_ptr, d_found, d_scores,
                  stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      // benchmark::read_from_ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op,
      //                          stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtFree(d_vectors_ptr));
      NPU_CHECK(aclrtFree(d_found));
      break;
    }
    case API_Select::find_or_insert_ptr: {
      V** d_vectors_ptr = nullptr;
      bool* d_found;
      NPU_CHECK(aclrtMalloc((void**)&d_found, key_num_per_op * sizeof(bool),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr, key_num_per_op * sizeof(V*),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      // benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op,
      //                      stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.start();
      table->find_or_insert(key_num_per_op, d_keys, d_vectors_ptr, d_found,
                            d_scores, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      NPU_CHECK(aclrtFree(d_vectors_ptr));
      NPU_CHECK(aclrtFree(d_found));
      break;
    }
    case API_Select::export_batch: {
      size_t* d_dump_counter;

      // Try to export close to but less than `key_num_per_op` data.
      // It's normal to happen `illegal memory access` error occasionally.
      float safe_ratio = 0.995;

      NPU_CHECK(aclrtMalloc((void**)&d_dump_counter, sizeof(size_t),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      NPU_CHECK(aclrtMemset(d_dump_counter, sizeof(size_t), 0, sizeof(size_t)));
      timer.start();
      table->export_batch(key_num_per_op / target_load_factor * safe_ratio, 0,
                          d_dump_counter, d_keys, d_vectors, d_scores, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      NPU_CHECK(aclrtFree(d_dump_counter));
      break;
    }
    case API_Select::export_batch_if: {
      size_t* d_dump_counter;

      // Try to export close to but less than `key_num_per_op` data.
      // It's normal to happen `illegal memory access` error occasionally.
      float safe_ratio = 0.995;

      NPU_CHECK(aclrtMalloc((void**)&d_dump_counter, sizeof(size_t),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      NPU_CHECK(aclrtMemset(d_dump_counter, sizeof(size_t), 0, sizeof(size_t)));
      timer.start();
      K pattern = 0;
      table->template export_batch_if<ExportIfPredFunctor>(
          pattern, threshold, key_num_per_op / target_load_factor *
          safe_ratio, 0, d_dump_counter, d_keys, d_vectors, d_scores,
          stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      NPU_CHECK(aclrtFree(d_dump_counter));
      break;
    }
    case API_Select::contains: {
      timer.start();
      table->contains(key_num_per_op, d_keys, d_found, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      break;
    }
    case API_Select::find_and_update: {
      V** d_vectors_ptr = nullptr;
      bool* d_found = nullptr;
      NPU_CHECK(aclrtMalloc((void**)&d_vectors_ptr, key_num_per_op * sizeof(V*),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      NPU_CHECK(aclrtMalloc((void**)&d_found, key_num_per_op * sizeof(bool),
                            ACL_MEM_MALLOC_HUGE_FIRST));
      // benchmark::array2ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.start();
      table->find_and_update(key_num_per_op, d_keys, d_vectors_ptr, d_found, d_scores, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      // benchmark::read_from_ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      NPU_CHECK(aclrtFree(d_vectors_ptr));
      NPU_CHECK(aclrtFree(d_found));
      break;
    }
    case API_Select::assign_scores: {
      timer.start();
      table->assign_scores(key_num_per_op, d_keys, d_scores, stream);
      NPU_CHECK(aclrtSynchronizeStream(stream));
      timer.end();
      break;
    }
    default: {
      std::cout << "[Unsupport API]\n";
    }
  }

  NPU_CHECK(aclrtDestroyStream(stream));

  NPU_CHECK(aclrtFreeHost(h_keys));
  NPU_CHECK(aclrtFreeHost(h_scores));
  NPU_CHECK(aclrtFreeHost(h_found));

  NPU_CHECK(aclrtFree(d_keys));
  NPU_CHECK(aclrtFree(d_scores_real));
  NPU_CHECK(aclrtFree(d_vectors));
  NPU_CHECK(aclrtFree(d_def_val));
  NPU_CHECK(aclrtFree(d_vectors_ptr));
  NPU_CHECK(aclrtFree(d_found));
  NPU_CHECK(aclrtFree(d_evict_keys));
  NPU_CHECK(aclrtFree(d_evict_scores));

  NPU_CHECK(aclrtSynchronizeDevice());
  NpuCheckError();

  float througput =
      key_num_per_op / timer.getResult() / (1024 * 1024 * 1024.0f);
  return througput;
}

void print_title_a() {
  cout << endl
       << "|    \u03BB "
       << "| insert_or_assign "
       << "| find_or_insert* "
       << "| find_and_update "
       << "|   find* "
       << "| export_batch "
       << "| assign_scores "
       << "| insert_and_evict ";
  cout << "|\n";

  //<< "| load_factor "
  cout << "|-----:"
       //<< "| insert_or_assign "
       << "|-----------------:"
       //<< "| find_or_insert* "
       << "|----------------:"
       //  << "| find_and_update "
       << "|----------------:"
       //<< "|   find* "
       << "|--------:"
       //<< "| export_batch "
       << "|-------------:"
       //<< "| assign_scores "
       << "|--------------:"
       //<< "| insert_and_evict "
       << "|-----------------:";
  cout << "|\n";
}

void print_title_b() {
  cout << endl
       << "|    \u03BB "
       << "| export_batch "
       << "| export_batch_if "
       << "|  contains "
       << "| find_and_update ";
  cout << "|\n";

  //<< "| load_factor "
  cout << "|-----:"
       //<< "| export_batch "
       << "|-------------:"
       //<< "| export_batch_if "
       << "|----------------:"
       //<< "|  contains "
       << "|----------:"
       //<< "| find_and_update "
       << "|----------------:";
  cout << "|\n";
}

void print_title_a_static() {
  cout << endl
       << "|    \u03BB "
       << "| find_or_insert ";
  cout << "|\n";

  //<< "| load_factor "
  cout << "|-----:"
       //<< "| find_or_insert "
       << "|---------------:";
  cout << "|\n";
}

void test_main(std::vector<API_Select>& apis, const size_t dim,
               const size_t init_capacity = 64 * 1024 * 1024UL,
               const size_t key_num_per_op = 1 * 1024 * 1024UL,
               const size_t hbm4values = 16, const float load_factor = 1.0f,
               const bool io_by_cpu = false,
               const std::vector<float> load_factors = {
                   0.50f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f, 1.00f}) {
  size_t free, total;
  NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));

  if (free / (1 << 30) < hbm4values) {
    std::cout << "free HBM is not enough, ignore current benchmark!"
              << std::endl;
    return;
  }
  TableOptions options;

  options.init_capacity = init_capacity;
  options.max_capacity = init_capacity;
  options.dim = dim;
  options.max_hbm_for_vectors = npu::hkv::GB(hbm4values);
  options.io_by_cpu = io_by_cpu;
  using Table = npu::hkv::HashTable<K, V, S, EvictStrategy::kLru>;

  std::shared_ptr<Table> table = std::make_shared<Table>();
  table->init(options);

  for (float load_factor : load_factors) {
    std::cout << "|" << rep(1) << fixed << setprecision(2) << load_factor
              << " ";

    for (auto api : apis) {
      table->clear();
      NPU_CHECK(aclrtSynchronizeDevice());
      // There is a sampling of load_factor after several times call to target
      // API. Two consecutive calls can avoid the impact of sampling.
      auto res1 = test_one_api<Table>(table, api, dim, init_capacity,
                                      key_num_per_op, load_factor);
      auto res2 = test_one_api<Table>(table, api, dim, init_capacity,
                                      key_num_per_op, load_factor);
      auto res = std::max(res1, res2);
      std::cout << "|";
      switch (api) {
        case API_Select::insert_or_assign: {
          // 空格数量为：列宽长度-7，如insert_or_assign的列宽长度为: insert_or_assign ,共18个字符，
          // 所以空格数量为18-7=11，具体列宽长度见print_title_a()
          std::cout << rep(11);
          break;
        }
        case API_Select::find_or_insert_ptr: {
          std::cout << rep(10);
          break;
        }
        case API_Select::find_and_update: {
          std::cout << rep(10);
          break;
        }
        case API_Select::find_ptr: {
          std::cout << rep(2);
          break;
        }
        case API_Select::export_batch: {
          std::cout << rep(7);
          break;
        }
        case API_Select::assign_scores: {
          std::cout << rep(8);
          break;
        }
        case API_Select::insert_and_evict: {
          std::cout << rep(11);
          break;
        }
        default: {
          std::cout << "[Unsupport API]";
        }
      }
      std::cout << fixed << setprecision(3) << setw(6) << setfill(' ') << res
                << " ";
    }
    std::cout << "|\n";
  }
}

void benchmark_hkv_hashtable(uint32_t block_dim) {
  size_t key_num_per_op = 1 * 1024 * 1024UL;
  cout << endl
       << "## Benchmark for hkv_hashtable" << endl
       << endl
       << "* block_dim: " << block_dim << endl
       << "* Key Type = uint64_t" << endl
       << "* Value Type = float32 * {dim}" << endl
       << "* Key-Values per OP = " << key_num_per_op << endl
       << "* Evict strategy: LRU" << endl
       << "* `\u03BB`"
       << ": load factor" << endl
       << "* `find*` means the `find` API that directly returns the addresses "
          "of values."
       << endl
       << "* `find_or_insert*` means the `find_or_insert` API that directly "
          "returns the addresses of values."
       << endl
       << "* ***Throughput Unit: Billion-KV/second***" << endl
       << endl;

  auto print_configuration = [](const size_t dim, const size_t init_capacity,
                                const size_t hbm4values) {
    using V = float;
    int32_t capacity = static_cast<int32_t>(init_capacity / (1024 * 1024));
    size_t hmem4values = init_capacity * dim * sizeof(V) / (1024 * 1024 * 1024);
    hmem4values = hmem4values < hbm4values ? 0 : (hmem4values - hbm4values);
    cout << "\n* dim = " << dim << ", "
         << "capacity = " << capacity << " Million-KV, "
         << "HBM = " << hbm4values << " GB, "
         << "HMEM = " << hmem4values << " GB\n";
  };
  try {
    {
      std::vector<API_Select> apis_a{
        API_Select::insert_or_assign,
        API_Select::find_or_insert_ptr,
        API_Select::find_and_update,
        API_Select::find_ptr,
        API_Select::export_batch,
        API_Select::assign_scores,
        API_Select::insert_and_evict,
      };
      cout << "### On pure HBM mode: " << endl;
      print_configuration(8, 128 * 1024 * 1024UL, 4);
      print_title_a();
      test_main(apis_a, 8, 128 * 1024 * 1024UL, key_num_per_op, 4);

      print_configuration(32, 128 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 32, 128 * 1024 * 1024UL, key_num_per_op, 16);

      print_configuration(64, 64 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 64, 64 * 1024 * 1024UL, key_num_per_op, 16);

      print_configuration(128, 32 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 128, 32 * 1024 * 1024UL, key_num_per_op, 16);

      print_configuration(256, 16 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 256, 16 * 1024 * 1024UL, key_num_per_op, 16);

      print_configuration(512, 8 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 512, 8 * 1024 * 1024UL, key_num_per_op, 16);

      print_configuration(1024, 4 * 1024 * 1024UL, 16);
      print_title_a();
      test_main(apis_a, 1024, 4 * 1024 * 1024UL, key_num_per_op, 16);

      cout << endl;
    }

    NPU_CHECK(aclrtSynchronizeDevice());
  } catch (const npu::hkv::NpuException& e) {
    cerr << e.what() << endl;
  }
  NPU_CHECK(aclrtSynchronizeDevice());
}

void query_memory() {
  size_t free = 0;
  size_t total = 0;
  NPU_CHECK(aclrtGetMemInfo(ACL_DDR_MEM, &free, &total));
  std::cout << "DDR free memory:" << free / (1 << 30)
            << " GB, DDR total memory:" << total / (1 << 30) << " GB"
            << std::endl;

  NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
  std::cout << "HBM free memory:" << free / (1 << 30)
            << " GB, HBM total memory:" << total / (1 << 30) << " GB"
            << std::endl;
}

int32_t main(int32_t argc, char* argv[]) {
  const char* socVersion = SOC_VERSION;
  auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  g_core_num_aiv = ascendc_platform->GetCoreNumAiv();
  std::cout << "Soc version: " << socVersion
            << " aiv_num: " << g_core_num_aiv << std::endl;

  NPU_CHECK(aclInit(nullptr));
  auto device_id_env = std::getenv("HKV_TEST_DEVICE");
  int32_t device_id = 0;
  try {
    device_id = (device_id_env != nullptr) ? std::stoi(device_id_env) : 0;
  } catch (...) {
    device_id = 0;
    std::cout << "set env HKV_TEST_DEVICE error, using default device_id 0" << std::endl;
  }
  NPU_CHECK(aclrtSetDevice(device_id));
  std::cout << "aclrtGetSocName:" << aclrtGetSocName() << " device_id:" << device_id << std::endl;
  query_memory();

  benchmark_hkv_hashtable(g_core_num_aiv);

  NPU_CHECK(aclrtResetDevice(device_id));
  NPU_CHECK(aclFinalize());
  return 0;
}
