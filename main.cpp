/**
 * @file main.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <assert.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "debug.h"
#include "hkv_hashtable.h"
#include "tests/test_util.h"
#include "tiling/platform/platform_ascendc.h"

using std::cerr;
using std::cout;
using std::endl;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

using namespace npu::hkv;

using K = uint64_t;
using S = uint64_t;
using V = float;
using EvictStrategy = npu::hkv::EvictStrategy;
using TableOptions = npu::hkv::HashTableOptions;

static uint32_t core_num_aiv = 0;

void demo_hkv_hashtable() {
  try {
    size_t key_num_per_op = 1 * 1 * 128UL;
    const size_t hbm4values = 16;
    const size_t init_capacity = 1 * 2 * 128UL;
    const size_t dim = 8;
    const bool io_by_cpu = false;

    size_t free, total;
    NPU_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &total));
    if (free / (1 << 30) < hbm4values) {
      std::cout << "free HBM is not enough, ignore current demo!" << std::endl;
      return;
    }
    TableOptions options;

    options.init_capacity = init_capacity;
    options.max_capacity = init_capacity;
    options.dim = dim;
    options.max_hbm_for_vectors = npu::hkv::GB(hbm4values);
    options.io_by_cpu = io_by_cpu;
    using Table = npu::hkv::HashTable<K, V, S, EvictStrategy::kCustomized>;

    std::shared_ptr<Table> table = std::make_shared<Table>();
    table->init(options);

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

    test_util::create_random_keys<K, S, V, dim>(h_keys, h_scores, h_vectors,
                                                key_num_per_op);

    K* d_keys;
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
    NPU_CHECK(aclrtMalloc((void**)&d_scores, key_num_per_op * sizeof(S),
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

    aclrtStream stream;
    NPU_CHECK(aclrtCreateStream(&stream));

    // initialize insert
    NPU_CHECK(aclrtMemcpy(d_keys, key_num_per_op * sizeof(K), h_keys,
                          key_num_per_op * sizeof(K),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_scores, key_num_per_op * sizeof(S), h_scores,
                          key_num_per_op * sizeof(S),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    NPU_CHECK(aclrtMemcpy(d_vectors, key_num_per_op * sizeof(V) * dim,
                          h_vectors, key_num_per_op * sizeof(V) * dim,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    table->find_or_insert(key_num_per_op, d_keys, d_vectors_ptr, d_found,
                          d_scores, stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));
    test_util::read_from_ptr(d_vectors_ptr, d_vectors, dim, key_num_per_op,
                             stream);
    NPU_CHECK(aclrtSynchronizeStream(stream));

    NPU_CHECK(aclrtMemcpy(h_found, key_num_per_op * sizeof(bool), d_found,
                          key_num_per_op * sizeof(bool),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtMemcpy(h_scores, key_num_per_op * sizeof(S), d_scores,
                          key_num_per_op * sizeof(S),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    NPU_CHECK(aclrtMemcpy(h_vectors, key_num_per_op * sizeof(V) * dim,
                          d_vectors, key_num_per_op * sizeof(V) * dim,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<void*> expect_values_ptr(key_num_per_op, nullptr);
    NPU_CHECK(aclrtMemcpy(
        expect_values_ptr.data(), key_num_per_op * sizeof(void*), d_vectors_ptr,
        key_num_per_op * sizeof(void*), ACL_MEMCPY_DEVICE_TO_HOST));

    size_t found_num = 0;
    size_t refused_num = 0;
    size_t insert_num = 0;
    for (size_t i = 0; i < key_num_per_op; i++) {
      if (h_found[i]) {
        found_num++;
      } else {
        if (expect_values_ptr[i] == nullptr) {
          refused_num++;
        } else {
          insert_num++;
        }
      }
      HKV_CHECK(h_scores[i] == h_keys[i], "score not equal key!");
    }
    HKV_CHECK(insert_num == key_num_per_op, "found num not equal key num!");

    NPU_CHECK(aclrtDestroyStream(stream));

    NPU_CHECK(aclrtFreeHost(h_keys));
    NPU_CHECK(aclrtFreeHost(h_scores));
    NPU_CHECK(aclrtFreeHost(h_vectors));
    NPU_CHECK(aclrtFreeHost(h_found));

    NPU_CHECK(aclrtFree(d_keys));
    NPU_CHECK(aclrtFree(d_scores));
    NPU_CHECK(aclrtFree(d_vectors));
    NPU_CHECK(aclrtFree(d_found));
    NPU_CHECK(aclrtFree(d_def_val));
    NPU_CHECK(aclrtFree(d_vectors_ptr));

    NPU_CHECK(aclrtFree(d_evict_keys));
    NPU_CHECK(aclrtFree(d_evict_scores));

    NPU_CHECK(aclrtSynchronizeDevice());
    NpuCheckError();
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
  uint32_t block_dim = 8;
  const char* soc_version = SOC_VERSION;
  auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(soc_version);
  HKV_CHECK(ascendc_platform != nullptr, "Get ascendc platform info failed, please check SOC_VERSION!");
  block_dim = ascendc_platform->GetCoreNumAiv();
  core_num_aiv = block_dim;
  std::cout << "Soc version: " << soc_version
            << " block_dim(aiv_num): " << block_dim << std::endl;

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
  std::cout << "aclrtGetSocName:" << aclrtGetSocName() << std::endl;
  query_memory();

  demo_hkv_hashtable();

  NPU_CHECK(aclrtResetDevice(device_id));
  NPU_CHECK(aclFinalize());
  std::cout << "HKV DEMO PASS!" << std::endl;
  return 0;
}