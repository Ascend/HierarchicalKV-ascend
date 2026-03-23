/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

/**
 * @file find_utils.h
 * @brief Device-side find helpers (digest probe). Included after `types.h`
 *        materializes `Bucket` / `OccupyResult`; depends on `ldg_l2nc_l1c` from
 *        `utils.h` via `types.h` include chain — do not include from `utils.h`.
 */
#pragma once

#include "types.h"

namespace npu {
namespace hkv {

template <class K, class V, class S>
__forceinline__ __device__ OccupyResult find_without_lock(
    __gm__ Bucket<K, V, S>* __restrict__ bucket,
    const K desired_key,
    uint32_t key_pos,
    const VecD_Comp target_digests,
    uint32_t& target_pos,
    const uint32_t bucket_capacity) {
  using BUCKET = Bucket<K, V, S>;
  constexpr uint32_t STRIDE = sizeof(VecD_Comp) / sizeof(D);

  VecD_Comp empty_digests_val = empty_digests<K>();
  for (uint32_t offset = 0; offset < bucket_capacity + STRIDE;
       offset += STRIDE) {
    uint32_t pos_cur = align_to<STRIDE>(key_pos);
    pos_cur = (pos_cur + offset) & (bucket_capacity - 1);

    __gm__ D* digests_ptr =
        BUCKET::digests(bucket->keys_, bucket_capacity, pos_cur);
    VecD_Comp probe_digests =
        ldg_l2nc_l1c(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));
    uint32_t possible_pos = 0;
    uint32_t cmp_result = vcmpeq4(probe_digests, target_digests);
    cmp_result &= 0x01010101;
    do {
      if (cmp_result == 0) {
        break;
      }
      uint32_t index =
          (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      possible_pos = pos_cur + index;
      const K current_key = ldg_l2nc_l1c(bucket->keys_ + possible_pos);
      if (current_key == desired_key) {
        target_pos = possible_pos;
        return OccupyResult::DUPLICATE;
      }
    } while (true);
    cmp_result = vcmpeq4(probe_digests, empty_digests_val);
    cmp_result &= 0x01010101;
    do {
      if (cmp_result == 0) {
        break;
      }
      uint32_t index =
          (AscendC::Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      possible_pos = pos_cur + index;
      if (offset == 0 && possible_pos < key_pos) {
        continue;
      }
      const K current_key = ldg_l2nc_l1c(bucket->keys_ + possible_pos);
      if (current_key == static_cast<K>(EMPTY_KEY)) {
        return OccupyResult::OCCUPIED_EMPTY;
      }
    } while (true);
  }
  return OccupyResult::CONTINUE;
}

}  // namespace hkv
}  // namespace npu
