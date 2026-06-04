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
#include <cstdlib>
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

  // Backward-compatible virtual interface (void**). New code should prefer the
  // templated overloads below, which avoid the `__gm__` address-space qualifier
  // drop warning when called with a `__gm__ T**`.
  virtual void alloc(const MemoryType type, void** ptr, size_t size,
                     unsigned int pinned_flags = 0) = 0;

  virtual void alloc_async(const MemoryType type, void** ptr, size_t size,
                           aclrtStream stream) = 0;

  virtual void free(const MemoryType type, void* ptr) = 0;

  virtual void free_async(const MemoryType type, void* ptr,
                          aclrtStream stream) = 0;

  // Templated wrappers: route through the virtual `void**` interface.
  // The pointer value is bit-copied from a `void*` to the caller's typed
  // pointer slot, which avoids an explicit `void*` -> `__gm__ T*` cast
  // (reinterpret_cast and static_cast are both rejected by the AscendC
  // compiler for this address-space conversion).
  template <typename T>
  void alloc(const MemoryType type, T** ptr, size_t size,
             unsigned int pinned_flags = 0) {
    void* tmp = nullptr;
    alloc(type, &tmp, size, pinned_flags);
    __builtin_memcpy(ptr, &tmp, sizeof(T*));
  }

  template <typename T>
  void alloc_async(const MemoryType type, T** ptr, size_t size,
                   aclrtStream stream) {
    void* tmp = nullptr;
    alloc_async(type, &tmp, size, stream);
    __builtin_memcpy(ptr, &tmp, sizeof(T*));
  }
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
        aclrtMallocAttrValue attrValue;
        attrValue.vaFlag = 1;

        aclrtMallocAttribute attribute[1];
        attribute[0].attr = ACL_RT_MEM_ATTR_VA_FLAG,
        attribute[0].value = attrValue;

        aclrtMallocConfig cfg;
        cfg.numAttrs = 1;
        cfg.attrs = attribute;
        NPU_CHECK(aclrtMallocHostWithCfg(ptr, size, &cfg));
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
