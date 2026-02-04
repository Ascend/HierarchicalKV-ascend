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

#include <stdlib.h>
#include "debug.h"

namespace npu {
namespace hkv {

enum MemoryType {
  Device,  // HBM
  Pinned,  // Pinned Host Memory
  Host,    // Host Memory
};

/* This abstract class defines the allocator APIs required by HKV.
   Any of the customized allocators should inherit from it.
 */
class BaseAllocator {
 public:
  BaseAllocator(const BaseAllocator&) = delete;
  BaseAllocator(BaseAllocator&&) = delete;

  BaseAllocator& operator=(const BaseAllocator&) = delete;
  BaseAllocator& operator=(BaseAllocator&&) = delete;

  BaseAllocator() = default;
  virtual ~BaseAllocator() = default;

  virtual void alloc(const MemoryType type, void** ptr, size_t size,
                     unsigned int pinned_flags = 0) = 0;

  virtual void alloc_async(const MemoryType type, void** ptr, size_t size,
                           aclrtStream stream) = 0;

  virtual void free(const MemoryType type, void* ptr) = 0;

  virtual void free_async(const MemoryType type, void* ptr,
                          aclrtStream stream) = 0;
};

class DefaultAllocator : public virtual BaseAllocator {
 public:
  DefaultAllocator() {};
  ~DefaultAllocator() override {};

  void alloc(const MemoryType type, void** ptr, size_t size,
             unsigned int pinned_flags = 0) override {
    switch (type) {
      case MemoryType::Device:
        NPU_CHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        break;
      case MemoryType::Pinned:
        NPU_CHECK(aclrtMallocHost(ptr, size));
        break;
      case MemoryType::Host:
        *ptr = std::malloc(size);
        break;
    }
    return;
  }

  void alloc_async(const MemoryType type, void** ptr, size_t size,
                   aclrtStream stream) override {
    if (type == MemoryType::Device) {
      NPU_CHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
      HKV_CHECK(false,
                "[DefaultAllocator] alloc_async is only support for "
                "MemoryType::Device!");
    }
    return;
  }

  void free(const MemoryType type, void* ptr) override {
    if (ptr == nullptr) {
      return;
    }
    switch (type) {
      case MemoryType::Pinned:
        NPU_CHECK(aclrtFreeHost(ptr));
        break;
      case MemoryType::Device:
        NPU_CHECK(aclrtFree(ptr));
        break;
      case MemoryType::Host:
        std::free(ptr);
        break;
    }
    return;
  }

  void free_async(const MemoryType type, void* ptr,
                  aclrtStream stream) override {
    if (ptr == nullptr) {
      return;
    }

    if (type == MemoryType::Device) {
      NPU_CHECK(aclrtFree(ptr));
    } else {
      HKV_CHECK(false,
                "[DefaultAllocator] free_async is only support for "
                "MemoryType::Device!");
    }
  }
};

}  // namespace hkv
}  // namespace npu
