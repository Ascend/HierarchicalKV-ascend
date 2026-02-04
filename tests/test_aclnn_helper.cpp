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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#define TEST_MEM
#include "aclnn_helper.h"
#include "test_util.h"

using namespace std;
using namespace npu::hkv;
using namespace testing;
using namespace test_util;

TEST(test_aclnn_helper, test_device_tensor) {
  // 1. 初始化
  init_env();

  // 2. 测试构造并释放内存
  auto dt = make_unique<DeviceTensor>();
  EXPECT_EQ(dt->get_shapes_data(), nullptr);
  EXPECT_EQ(dt->get_shapes_size(), 0);
  EXPECT_EQ(dt->get_data_type(), aclDataType::ACL_DT_UNDEFINED);
  EXPECT_EQ(dt->get_data(), nullptr);
  EXPECT_EQ(dt->get_data_size(), 0);

  EXPECT_THROW(
      { dt->init(aclDataType::ACL_DT_UNDEFINED, {1}); }, runtime_error);
  EXPECT_THROW({ dt->init(aclDataType::ACL_UINT64, {2}); }, runtime_error);
  EXPECT_EQ(dt->get_data(), nullptr);

  dt->init(aclDataType::ACL_INT32, {3, 4});
  EXPECT_NE(dt->get_data(), nullptr);
  EXPECT_EQ(dt->get_shapes_data()[0], 3);
  EXPECT_EQ(dt->get_shapes_data()[1], 4);

  dt = make_unique<DeviceTensor>();
  dt->init(aclDataType::ACL_INT64, {4, 7});
  EXPECT_NE(dt->get_data(), nullptr);
  EXPECT_EQ(dt->get_shapes_size(), 2);
  EXPECT_EQ(dt->get_data_size(), 4 * 7 * sizeof(int64_t));

  dt = make_unique<DeviceTensor>();
  dt->init(aclDataType::ACL_FLOAT, {5, 1});
  EXPECT_NE(dt->get_data(), nullptr);
  EXPECT_EQ(dt->get_data_type(), aclDataType::ACL_FLOAT);

  // 3. 测试不释放内存
  auto dt2 = make_unique<DeviceTensor>();
  EXPECT_THROW(
      { dt2->init(dt->get_data(), aclDataType::ACL_INT8, {23, 2}); },
      runtime_error);
  dt2->init(dt->get_data(), aclDataType::ACL_FLOAT, {77});
  EXPECT_EQ(dt2->get_shapes_data()[0], 77);
  EXPECT_EQ(dt2->get_shapes_size(), 1);
  EXPECT_EQ(dt2->get_data_type(), aclDataType::ACL_FLOAT);
  EXPECT_EQ(dt2->get_data(), dt->get_data());
  EXPECT_EQ(dt2->get_data_size(), 77 * sizeof(float));

  // 4. 保证内存释放
  dt.reset();
  dt2.reset();
  EXPECT_TRUE(g_mem_set.empty());
}

class AclnnMockInterface {
 public:
  virtual ~AclnnMockInterface() = default;

  virtual int aclnnMockGetWorkspaceSize(aclTensor* in1, aclIntArray* in2,
                                        bool in3, aclDataType in4,
                                        aclTensor* in5, uint64_t* workspaceSize,
                                        aclOpExecutor** executor) = 0;

  virtual int aclnnMock(void* workspace, uint64_t workspaceSize,
                        aclOpExecutor* executor, aclrtStream stream) = 0;
};

class AclnnMock : public AclnnMockInterface {
 public:
  MOCK_METHOD(int, aclnnMockGetWorkspaceSize,
              (aclTensor * in1, aclIntArray* in2, bool in3, aclDataType in4,
               aclTensor* in5, uint64_t* workspaceSize,
               aclOpExecutor** executor),
              (override));
  MOCK_METHOD(int, aclnnMock,
              (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
               aclrtStream stream),
              (override));
};

static unique_ptr<AclnnMock> g_aclnn_mock;

int aclnnMockGetWorkspaceSize(aclTensor* in1, aclIntArray* in2, bool in3,
                              aclDataType in4, aclTensor* in5,
                              uint64_t* workspaceSize,
                              aclOpExecutor** executor) {
  return g_aclnn_mock->aclnnMockGetWorkspaceSize(in1, in2, in3, in4, in5,
                                          workspaceSize, executor);
}

int aclnnMock(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
              aclrtStream stream) {
  return g_aclnn_mock->aclnnMock(workspace, workspaceSize, executor, stream);
}

TEST(test_aclnn_helper, test_exec_aclnn_op_success) {
  // 1. 初始化
  init_env();

  // 2. 构造mock对象
  g_aclnn_mock = make_unique<AclnnMock>();
  EXPECT_CALL(*g_aclnn_mock, aclnnMockGetWorkspaceSize)
      .Times(1)
      .WillOnce(Return(ACL_SUCCESS));
  EXPECT_CALL(*g_aclnn_mock, aclnnMock).Times(1).WillOnce(Return(ACL_SUCCESS));

  // 3. 执行
  auto in = make_unique<DeviceTensor>();
  in->init(aclDataType::ACL_INT32, {1024});
  vector<int64_t> dims = {0};
  bool keep_dims = false;
  auto out_data_type = aclDataType::ACL_INT64;
  auto out = make_unique<DeviceTensor>();
  out->init(out_data_type, {1});
  aclrtStream stream = 0;

  EXEC_ACLNN_OP(aclnnMock, *in, dims, keep_dims, out_data_type, *out);
  g_aclnn_mock.reset();

  in.reset();
  out.reset();
  EXPECT_TRUE(g_mem_set.empty());
}

TEST(test_aclnn_helper, test_exec_aclnn_op_get_workspace_size_fail) {
  // 1. 初始化
  init_env();

  // 2. 构造mock对象
  g_aclnn_mock = make_unique<AclnnMock>();
  EXPECT_CALL(*g_aclnn_mock, aclnnMockGetWorkspaceSize)
      .Times(1)
      .WillOnce(Return(ACL_ERROR_INVALID_PARAM));
  EXPECT_CALL(*g_aclnn_mock, aclnnMock).Times(0);

  // 3. 执行
  auto in = make_unique<DeviceTensor>();
  in->init(aclDataType::ACL_INT32, {111});
  vector<int64_t> dims = {0};
  bool keep_dims = false;
  auto out_data_type = aclDataType::ACL_INT64;
  auto out = make_unique<DeviceTensor>();
  out->init(out_data_type, {1});
  aclrtStream stream = 0;

  EXPECT_THROW(
      { EXEC_ACLNN_OP(aclnnMock, *in, dims, keep_dims, out_data_type, *out); },
      runtime_error);
  g_aclnn_mock.reset();

  in.reset();
  out.reset();
  EXPECT_TRUE(g_mem_set.empty());
}

TEST(test_aclnn_helper, test_exec_aclnn_op_fail) {
  // 1. 初始化
  init_env();

  // 2. 构造mock对象
  g_aclnn_mock = make_unique<AclnnMock>();
  EXPECT_CALL(*g_aclnn_mock, aclnnMockGetWorkspaceSize)
      .Times(1)
      .WillOnce(Return(ACL_SUCCESS));
  EXPECT_CALL(*g_aclnn_mock, aclnnMock).Times(1).WillOnce(Return(ACL_ERROR_WRITE_FILE));

  // 3. 执行
  auto in = make_unique<DeviceTensor>();
  in->init(aclDataType::ACL_INT32, {123});
  vector<int64_t> dims = {0};
  bool keep_dims = false;
  auto out_data_type = aclDataType::ACL_INT64;
  auto out = make_unique<DeviceTensor>();
  out->init(out_data_type, {55});
  aclrtStream stream = 0;

  EXPECT_THROW(
      { EXEC_ACLNN_OP(aclnnMock, *in, dims, keep_dims, out_data_type, *out); },
      runtime_error);
  g_aclnn_mock.reset();

  in.reset();
  out.reset();
  EXPECT_TRUE(g_mem_set.empty());
}
