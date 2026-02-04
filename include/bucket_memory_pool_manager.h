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

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <acl/acl.h>
#include "allocator.h"
#include "hashtable_options.h"
#include "table.h"
#include "types.h"
#include "utils.h"

namespace npu {
namespace hkv {
constexpr size_t CACHE_LINE_SIZE = 128U / sizeof(uint8_t);
constexpr size_t BUCKET_ALIGN_SIZE = 512;

/**
 * @brief Manages bucket memory: either a single pre-allocated pool (when enabled)
 * or per-block allocations via allocator. Implements IBucketAddressProvider so
 * that create_table / initialize_buckets / double_capacity use it for all
 * bucket address and size queries.
 */
template <class K, class V, class S>
class BucketMemoryPoolManager : public IBucketAddressProvider {
 public:
  using key_type = K;
  using value_type = V;
  using score_type = S;

  BucketMemoryPoolManager() = default;
  ~BucketMemoryPoolManager() override {
    if (use_bucket_memory_pool_ && bucket_memory_pool_ != nullptr) {
      NPU_CHECK(aclrtFree(bucket_memory_pool_));
      bucket_memory_pool_ = nullptr;
      bucket_memory_pool_size_ = 0;
      bucket_memory_size_ = 0;
    }
    if (!use_bucket_memory_pool_ && allocator_ != nullptr) {
      for (uint8_t* block : block_bases_) {
        if (block != nullptr) {
          allocator_->free(MemoryType::Device, block);
        }
      }
      block_bases_.clear();
      allocator_ = nullptr;
    }
  }

  void initialize(const HashTableOptions& options) {
    options_ = options;
    parse_memory_pool_config();
    if (use_bucket_memory_pool_) {
      allocate_bucket_memory_pool();
    } else {
      bucket_memory_pool_ = nullptr;
      bucket_memory_pool_size_ = 0;
      bucket_memory_size_ = 0;
      bucket_memory_size_non_pool_ = calculate_bucket_memory_size_non_pool();
    }
  }

  void ensure_buckets_for_range(size_t start, size_t end,
                                size_t num_of_buckets_per_alloc,
                                BaseAllocator* allocator) override {
    if (use_bucket_memory_pool_) {
      return;  // no-op, pool already covers
    }
    if (allocator != nullptr) {
      allocator_ = allocator;
    }
    if (num_of_buckets_per_alloc_ == 0) {
      num_of_buckets_per_alloc_ = num_of_buckets_per_alloc;
    }
    size_t bucket_memory_size = get_bucket_memory_size();
    for (size_t i = start; i < end; i += num_of_buckets_per_alloc) {
      size_t num_of_buckets =
          std::min(end - i, static_cast<size_t>(num_of_buckets_per_alloc));
      uint8_t* address = nullptr;
      allocator->alloc(MemoryType::Device, (void**)&address,
                       bucket_memory_size * num_of_buckets);
      block_bases_.push_back(address);
    }
  }

  uint8_t* get_bucket_address(size_t bucket_index) const override {
    if (use_bucket_memory_pool_) {
      return bucket_memory_pool_ + bucket_index * bucket_memory_size_;
    }
    if (block_bases_.empty() || num_of_buckets_per_alloc_ == 0) {
      return nullptr;
    }
    size_t block_index = bucket_index / num_of_buckets_per_alloc_;
    size_t offset_in_block = bucket_index % num_of_buckets_per_alloc_;
    return block_bases_[block_index] +
           offset_in_block * bucket_memory_size_non_pool_;
  }

  size_t get_bucket_memory_size() const override {
    return use_bucket_memory_pool_ ? bucket_memory_size_
                                   : bucket_memory_size_non_pool_;
  }

  bool use_pool() const override { return use_bucket_memory_pool_; }

  void ensure_capacity(size_t new_buckets_num) {
    if (!use_bucket_memory_pool_) {
      return;
    }
    size_t required_size = bucket_memory_size_ * new_buckets_num;
    if (bucket_memory_pool_ == nullptr || required_size <= bucket_memory_pool_size_) {
      return;
    }
    size_t max_buckets_num = calculate_max_buckets_num();
    size_t new_pool_size = bucket_memory_size_ * max_buckets_num;
    uint8_t* new_bucket_memory_pool = nullptr;
    NPU_CHECK(aclrtMalloc((void**)&new_bucket_memory_pool, new_pool_size,
                          bucket_malloc_flag_));
    NPU_CHECK(aclrtFree(bucket_memory_pool_));
    bucket_memory_pool_ = new_bucket_memory_pool;
    bucket_memory_pool_size_ = new_pool_size;
  }

 private:
  bool parse_key_value_pair(const std::string& pair, std::string& key,
                            std::string& value) const {
    size_t pos = pair.find('=');
    if (pos == std::string::npos) return false;
    key = pair.substr(0, pos);
    value = pair.substr(pos + 1);
    size_t key_start = key.find_first_not_of(" \t");
    if (key_start != std::string::npos) {
      key = key.substr(key_start);
      size_t key_end = key.find_last_not_of(" \t");
      if (key_end != std::string::npos) {
        key = key.substr(0, key_end + 1);
      }
    }
    size_t value_start = value.find_first_not_of(" \t");
    if (value_start != std::string::npos) {
      value = value.substr(value_start);
      size_t value_end = value.find_last_not_of(" \t");
      if (value_end != std::string::npos) {
        value = value.substr(0, value_end + 1);
      }
    }
    return true;
  }

  void parse_environment_config(const std::string& env_str,
                                std::string& buckets_mem_pool_value,
                                std::string& page_table_value) const {
    buckets_mem_pool_value.clear();
    page_table_value.clear();
    size_t start = 0;
    size_t pos = 0;
    while ((pos = env_str.find(';', start)) != std::string::npos) {
      std::string pair = env_str.substr(start, pos - start);
      std::string key, value;
      if (parse_key_value_pair(pair, key, value)) {
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        if (key == "buckets_mem_pool") {
          buckets_mem_pool_value = value;
        } else if (key == "page_table") {
          page_table_value = value;
        }
      }
      start = pos + 1;
    }
    if (start < env_str.length()) {
      std::string pair = env_str.substr(start);
      std::string key, value;
      if (parse_key_value_pair(pair, key, value)) {
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        if (key == "buckets_mem_pool") {
          buckets_mem_pool_value = value;
        } else if (key == "page_table") {
          page_table_value = value;
        }
      }
    }
  }

  void apply_memory_pool_config(const std::string& buckets_mem_pool_value,
                                const std::string& page_table_value) {
    if (buckets_mem_pool_value == "disable") {
      use_bucket_memory_pool_ = false;
      bucket_malloc_flag_ = ACL_MEM_MALLOC_HUGE_FIRST;
    } else if (buckets_mem_pool_value == "enable") {
      use_bucket_memory_pool_ = true;
      if (page_table_value == "1g") {
        bucket_malloc_flag_ = ACL_MEM_MALLOC_HUGE1G_ONLY;
      } else if (page_table_value == "2m") {
        bucket_malloc_flag_ = ACL_MEM_MALLOC_HUGE_ONLY;
      }
    }
  }

  void parse_memory_pool_config() {
    use_bucket_memory_pool_ = true;
    bucket_malloc_flag_ = ACL_MEM_MALLOC_HUGE_ONLY;
    const char* env_alloc_conf = std::getenv("HKV_NPU_ALLOC_CONF");
    if (env_alloc_conf != nullptr) {
      std::string alloc_conf(env_alloc_conf);
      std::string buckets_mem_pool_value;
      std::string page_table_value;
      parse_environment_config(alloc_conf, buckets_mem_pool_value,
                               page_table_value);
      apply_memory_pool_config(buckets_mem_pool_value, page_table_value);
    }
  }

  size_t calculate_bucket_memory_size() const {
    size_t bucket_memory_size =
        options_.max_bucket_size * (sizeof(key_type) + sizeof(score_type));
    size_t reserve_size = options_.max_bucket_size < CACHE_LINE_SIZE
                                ? CACHE_LINE_SIZE
                                : options_.max_bucket_size;
    bucket_memory_size += reserve_size * sizeof(uint8_t);
    return (bucket_memory_size + BUCKET_ALIGN_SIZE - 1) / BUCKET_ALIGN_SIZE * BUCKET_ALIGN_SIZE;
  }

  size_t calculate_bucket_memory_size_non_pool() const {
    size_t bucket_memory_size =
        options_.max_bucket_size * (sizeof(key_type) + sizeof(score_type));
    size_t reserve_size = options_.max_bucket_size < CACHE_LINE_SIZE
                                ? CACHE_LINE_SIZE
                                : options_.max_bucket_size;
    bucket_memory_size += reserve_size * sizeof(uint8_t);
    return bucket_memory_size;
  }

  size_t calculate_max_buckets_num() const {
    if ((options_.init_capacity * 2) > options_.max_capacity) {
      return 1 + ((options_.max_capacity - 1) / options_.max_bucket_size);
    }
    size_t max_buckets_num = 1;
    size_t temp_capacity = options_.max_bucket_size;
    while (temp_capacity < options_.max_capacity) {
      max_buckets_num *= 2;
      temp_capacity = max_buckets_num * options_.max_bucket_size;
    }
    return max_buckets_num;
  }

  void allocate_bucket_memory_pool() {
    bucket_memory_size_ = calculate_bucket_memory_size();
    size_t max_buckets_num = calculate_max_buckets_num();
    bucket_memory_pool_size_ = bucket_memory_size_ * max_buckets_num;
    NPU_CHECK(aclrtMalloc((void**)&bucket_memory_pool_, bucket_memory_pool_size_,
                          bucket_malloc_flag_));
  }

  HashTableOptions options_;
  uint8_t* bucket_memory_pool_ = nullptr;
  size_t bucket_memory_pool_size_ = 0;
  size_t bucket_memory_size_ = 0;
  bool use_bucket_memory_pool_ = false;
  aclrtMemMallocPolicy bucket_malloc_flag_ = ACL_MEM_MALLOC_HUGE_FIRST;

  std::vector<uint8_t*> block_bases_;
  size_t bucket_memory_size_non_pool_ = 0;
  BaseAllocator* allocator_ = nullptr;
  size_t num_of_buckets_per_alloc_ = 0;
};

}  // namespace hkv
}  // namespace npu
