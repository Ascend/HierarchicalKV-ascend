/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <mutex>
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"

namespace npu {
namespace hkv {

struct ValueMoveTiling {
  uint32_t former_num;
  uint64_t former_core_move_num;
  uint64_t tail_core_move_num;
  uint64_t valid_ub_size;
  uint32_t tile_size;
  uint32_t num_tiles;
};

static uint64_t GetTotalUbSize() {
  static std::once_flag init_flag;
  static uint64_t ub_total_size = 0;
  std::call_once(init_flag, []() {
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    HKV_CHECK(platform != nullptr, "get platform failed.");
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
  });
  return ub_total_size;
}

static uint64_t GetMixedOpUbSize() {
  static std::once_flag init_flag;
  static uint64_t valid_ub_size = 0;
  std::call_once(init_flag, []() {
    const uint64_t RESERVE_UB_SIZE = KB(8);
    const uint64_t SIMT_UB_SIZE = KB(32);
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    HKV_CHECK(platform != nullptr, "get platform failed.");
    uint64_t ub_total_size = GetTotalUbSize();
    HKV_CHECK(ub_total_size > RESERVE_UB_SIZE + SIMT_UB_SIZE,
              log_format("UB size %lu is too small.", ub_total_size));
    valid_ub_size = ub_total_size - RESERVE_UB_SIZE - SIMT_UB_SIZE;
  });
  return valid_ub_size;
}

static inline ValueMoveTiling GetValueMoveTiling(uint32_t n, uint32_t block_dim,
                                                 uint32_t dim,
                                                 uint32_t element_size,
                                                 bool is_pure_simd, uint32_t buffer_num = DOUBLE_BUFFER) {
  ValueMoveTiling tiling;
  tiling.tail_core_move_num = n / block_dim;
  tiling.former_core_move_num = tiling.tail_core_move_num + 1;
  tiling.former_num = n - tiling.tail_core_move_num * block_dim;
  tiling.valid_ub_size = is_pure_simd ? GetTotalUbSize() : GetMixedOpUbSize();

  uint32_t max_tile_size =
      tiling.valid_ub_size / (buffer_num * element_size);
  HKV_CHECK(max_tile_size != 0,
            log_format("UB size %lu is too small.", tiling.valid_ub_size));
  tiling.tile_size = (dim <= max_tile_size) ? dim : max_tile_size;
  tiling.num_tiles = (dim + tiling.tile_size - 1) / tiling.tile_size;

  return tiling;
}

}  // namespace hkv
}  // namespace npu
