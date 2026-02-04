/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef ASCENDC_EVICT_STRATEGY_DISPATCHER_H_
#define ASCENDC_EVICT_STRATEGY_DISPATCHER_H_

#include "types.h"

namespace npu {
namespace hkv {

enum DataSize { BYTE_1 = 1, BYTE_2 = 2, BYTE_4 = 4, BYTE_8 = 8, BYTE_16 = 16 };

/**
 * @brief 通用的数据类型分发宏
 * 用法示例：
 * DISPATCH_VALUE_SIZE(value_size,
 *   (Simt::VF_CALL<kernel_vf<K, DTYPE, S, STRATEGY>>(args...);
 * ));
 */
#define DISPATCH_VALUE_SIZE(value_size, FUNC) \
  do {                                        \
    switch (value_size) {                     \
      case BYTE_1: {                          \
        using DTYPE = int8_t;                 \
        FUNC;                                 \
        break;                                \
      }                                       \
      case BYTE_2: {                          \
        using DTYPE = int16_t;                \
        FUNC;                                 \
        break;                                \
      }                                       \
      case BYTE_4: {                          \
        using DTYPE = int32_t;                \
        FUNC;                                 \
        break;                                \
      }                                       \
      case BYTE_8: {                          \
        using DTYPE = int64_t;                \
        FUNC;                                 \
        break;                                \
      }                                       \
      case BYTE_16: {                         \
        using DTYPE = int4;                   \
        FUNC;                                 \
        break;                                \
      }                                       \
      default:                                \
        break;                                \
    }                                         \
  } while (0)

#define DISPATCH_GROUP_SIZE(group_size, FUNC) \
  do {                                        \
    switch (group_size) {                     \
      case 2: {                               \
        constexpr int32_t GROUP_SIZE = 2;     \
        FUNC;                                 \
        break;                                \
      }                                       \
      case 4: {                               \
        constexpr int32_t GROUP_SIZE = 4;     \
        FUNC;                                 \
        break;                                \
      }                                       \
      case 8: {                               \
        constexpr int32_t GROUP_SIZE = 8;     \
        FUNC;                                 \
        break;                                \
      }                                       \
      case 16: {                              \
        constexpr int32_t GROUP_SIZE = 16;    \
        FUNC;                                 \
        break;                                \
      }                                       \
      case 32: {                              \
        constexpr int32_t GROUP_SIZE = 32;    \
        FUNC;                                 \
        break;                                \
      }                                       \
    }                                         \
  } while (0)

/**
 * @brief 通用的淘汰策略分发宏
 *
 * 用法示例：
 *
 * DISPATCH_EVICT_STRATEGY(evict_strategy, [&] {
 *   Simt::VF_CALL<kernel_vf<K, V, S, STRATEGY>>(args...);
 * });
 *
 */
#define DISPATCH_EVICT_STRATEGY(runtime_strategy, FUNC)                        \
  do {                                                                         \
    switch (runtime_strategy) {                                                \
      case npu::hkv::EvictStrategyInternal::kLru: {                            \
        constexpr int STRATEGY = npu::hkv::EvictStrategyInternal::kLru;        \
        FUNC;                                                                  \
        break;                                                                 \
      }                                                                        \
      case npu::hkv::EvictStrategyInternal::kLfu: {                            \
        constexpr int STRATEGY = npu::hkv::EvictStrategyInternal::kLfu;        \
        FUNC;                                                                  \
        break;                                                                 \
      }                                                                        \
      case npu::hkv::EvictStrategyInternal::kEpochLru: {                       \
        constexpr int STRATEGY = npu::hkv::EvictStrategyInternal::kEpochLru;   \
        FUNC;                                                                  \
        break;                                                                 \
      }                                                                        \
      case npu::hkv::EvictStrategyInternal::kEpochLfu: {                       \
        constexpr int STRATEGY = npu::hkv::EvictStrategyInternal::kEpochLfu;   \
        FUNC;                                                                  \
        break;                                                                 \
      }                                                                        \
      case npu::hkv::EvictStrategyInternal::kCustomized: {                     \
        constexpr int STRATEGY = npu::hkv::EvictStrategyInternal::kCustomized; \
        FUNC;                                                                  \
        break;                                                                 \
      }                                                                        \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
  } while (0)

}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_EVICT_STRATEGY_DISPATCHER_H_
