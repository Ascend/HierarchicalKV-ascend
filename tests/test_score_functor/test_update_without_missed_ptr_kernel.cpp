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

#include "./v35/test_score_functor_kernel.h"
#include "kernel_operator.h"

using namespace npu::hkv;

extern "C" __global__ __aicore__ void test_update_without_missed_ptr_kernel(
    int strategy, GM_ADDR bucket_keys, uint32_t bucket_capacity, uint32_t key_pos,
    GM_ADDR input_scores, int key_idx, uint64_t epoch, uint64_t cur_cycle) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  DISPATCH_STRATEGY(test_update_without_missed_ptr_vf, bucket_keys, bucket_capacity,
                    key_pos, input_scores, key_idx, epoch, cur_cycle);
}
