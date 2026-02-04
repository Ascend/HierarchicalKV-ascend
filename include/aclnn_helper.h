/**
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

#pragma once

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <tuple>
#include <utility>
#include <vector>
#include "debug.h"

#ifdef TEST_MEM
#include <unordered_set>
namespace {
std::unordered_set<void*> g_mem_set;
#define CHECK_MEM_ALLOC(p)                                             \
  do {                                                                 \
    HKV_CHECK((g_mem_set.find(p) == g_mem_set.end()), "alloc error!"); \
    g_mem_set.insert(p);                                               \
  } while (false)

#define CHECK_MEM_RELEASE(p)                                             \
  do {                                                                   \
    HKV_CHECK((g_mem_set.find(p) != g_mem_set.end()), "release error!"); \
    g_mem_set.erase(p);                                                  \
  } while (false)
}  // namespace
#else
#define CHECK_MEM_ALLOC(p) (void)(p)
#define CHECK_MEM_RELEASE(p) (void)(p)
#endif

namespace npu {
namespace hkv {

inline size_t get_acl_data_type_size(aclDataType d_type) {
  switch (d_type) {
    case aclDataType::ACL_INT32: {
      return sizeof(int32_t);
    }
    case aclDataType::ACL_INT64: {
      return sizeof(int64_t);
    }
    case aclDataType::ACL_FLOAT: {
      return sizeof(float);
    }
    default: {
      HKV_CHECK(false, log_format("d_type {} is not supported.", d_type));
    }
  }

  return 0;
}

inline size_t get_shape_size(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return static_cast<size_t>(shape_size);
}

/*
 * DeviceTensor是对device内存的封装，便于后续转为aclTensor
 * 考虑到已申请内存仅封装以及需申请内存并封装的两种场景提供了两个init接口
 * 已申请内存的数据shape和长度由用户自行保证
 */
class DeviceTensor {
 public:
  DeviceTensor() = default;

  void init(void* in_data, aclDataType in_d_type,
            std::vector<int64_t>&& in_shapes) {
    data = in_data;
    d_type = in_d_type;
    shapes = std::move(in_shapes);
    data_size = get_acl_data_type_size(d_type) * get_shape_size(shapes);
    clear_flag = false;
  }

  void init(aclDataType in_type, std::vector<int64_t>&& in_shapes) {
    if (data != nullptr) {
      return;
    }

    data_size = get_acl_data_type_size(in_type) * get_shape_size(in_shapes);
    auto ret = aclrtMalloc(&data, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    NPU_CHECK(ret);
    CHECK_MEM_ALLOC(data);
    d_type = in_type;
    shapes = in_shapes;
    clear_flag = true;
  }

  ~DeviceTensor() {
    if (clear_flag && (data != nullptr)) {
      (void)aclrtFree(data);
      CHECK_MEM_RELEASE(data);
    }
  }

  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  const int64_t* get_shapes_data() const { return shapes.data(); }

  size_t get_shapes_size() const { return shapes.size(); }

  aclDataType get_data_type() const { return d_type; }

  void* get_data() const { return data; }

  size_t get_data_size() const { return data_size; }

 private:
  void* data = nullptr;
  aclDataType d_type = aclDataType::ACL_DT_UNDEFINED;
  std::vector<int64_t> shapes;
  size_t data_size = 0;
  // 默认不管理生命周期
  bool clear_flag = false;
};

inline aclTensor* convert_type(const DeviceTensor& tensor) {
  auto shape = tensor.get_shapes_data();
  auto shape_size = tensor.get_shapes_size();
  std::vector<int64_t> strides(shape_size, 1);
  for (int64_t i = static_cast<int64_t>(shape_size) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  auto target_tensor = aclCreateTensor(
      shape, shape_size, tensor.get_data_type(), strides.data(), 0,
      aclFormat::ACL_FORMAT_ND, shape, shape_size, tensor.get_data());
  HKV_CHECK((target_tensor != nullptr), "aclCreateTensor error.");
  CHECK_MEM_ALLOC(target_tensor);
  return target_tensor;
}

inline aclIntArray* convert_type(const std::vector<int64_t>& vec) {
  if (vec.empty()) {
    return nullptr;
  }

  auto array = aclCreateIntArray(vec.data(), vec.size());
  HKV_CHECK((array != nullptr), "aclCreateIntArray error.");
  CHECK_MEM_ALLOC(array);
  return array;
}

template <typename T>
T convert_type(T t) {
  return t;
}

inline void release(aclTensor* p) {
  if (p != nullptr) {
    (void)aclDestroyTensor(p);
    CHECK_MEM_RELEASE(p);
  }
}

inline void release(aclIntArray* p) {
  if (p != nullptr) {
    (void)aclDestroyIntArray(p);
    CHECK_MEM_RELEASE(p);
  }
}

template <typename T>
void release(T) {}

template <typename Tuple, size_t... I>
void call_release(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void release_convert_types(Tuple& t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  call_release(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto convert_types(Ts&... args) {
  return std::make_tuple(convert_type(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

template <typename... Args>
class TupleDeleter {
 public:
  explicit TupleDeleter(std::tuple<Args...>& t) : t(t) {};

  ~TupleDeleter() { release_convert_types(t); }

  TupleDeleter(const TupleDeleter&) = delete;
  TupleDeleter& operator=(const TupleDeleter&) = delete;

 private:
  std::tuple<Args...>& t;
};

class WorkspaceDeleter {
 public:
  WorkspaceDeleter() = default;

  ~WorkspaceDeleter() {
    if (workspace != nullptr) {
      (void)aclrtFree(workspace);
      CHECK_MEM_RELEASE(workspace);
    }
  }

  void set_workspace(void* workspace) { this->workspace = workspace; }

  WorkspaceDeleter(const WorkspaceDeleter&) = delete;
  WorkspaceDeleter& operator=(const WorkspaceDeleter&) = delete;

 private:
  void* workspace = nullptr;
};

class ScopedStream {
  public:
    explicit ScopedStream(aclrtStream external_stream) : stream_(external_stream), owned_(false) {
      if (stream_ == nullptr) {
        auto ret = aclrtCreateStream(&stream_);
        NPU_CHECK(ret);
        owned_ = true;
      }
    }
 
    ~ScopedStream() {
      if (owned_ && stream_ != nullptr) {
        auto ret = aclrtDestroyStream(stream_);
        NPU_CHECK(ret);
      }
    }
 
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;
  
    aclrtStream get() const {
      return stream_;
    }
 
  private:
    aclrtStream stream_;
    bool owned_;
};

#define GET_WORKSPACE_SIZE_FUNC(aclnn_api) aclnn_api##GetWorkspaceSize

/*
 * EXEC_ACLNN_OP使用示例：
 * EXEC_ACLNN_OP(aclnnReduceSum, input, dims, keep_dims, out_data_type, out);
 * 其中第一个参数为aclnn op名称，后续参数为其对应输入
 * EXEC_ACLNN_OP通过字符串拼接得到aclnnXXGetWorkspaceSize接口和aclnnXX接口，
 * aclnnXXGetWorkspaceSize接口参数数量不固定且需要转为特定入参（aclTensor等），因此使用了tuple将参数进行打包
 * aclnnXX接口参数数量固定，因此使用直调模式。
 * convert_types负责将参数转为特定入参（DeviceTensor转aclTensor等），并将参数打包成一个tuple对象。
 * release_convert_types负责结束后释放convert_type过程中申请的内存。
 * 因此如需【新增转换类型】，则应添加对应模板特例化的convert_type接口、release接口。
 * 此外，由于接口可能失败导致异常，为保证内存得到合理释放，新增TupleDeleter、WorkspaceDeleter类用于在各种场景下释放内存
 */
#define EXEC_ACLNN_OP(aclnn_api, ...)                                   \
  do {                                                                  \
    uint64_t workspace_size = 0;                                        \
    uint64_t* workspace_size_addr = &workspace_size;                    \
    aclOpExecutor* executor = nullptr;                                  \
    aclOpExecutor** executor_addr = &executor;                          \
    auto converted_params =                                             \
        convert_types(__VA_ARGS__, workspace_size_addr, executor_addr); \
    TupleDeleter tuple_deleter(converted_params);                       \
    auto workspace_status =                                             \
        call(GET_WORKSPACE_SIZE_FUNC(aclnn_api), converted_params);     \
    NPU_CHECK(workspace_status);                                        \
    WorkspaceDeleter workspace_deleter;                                 \
    void* workspace_addr = nullptr;                                     \
    if (workspace_size != 0) {                                          \
      auto aclnn_ret = aclrtMalloc(&workspace_addr, workspace_size,     \
                                   ACL_MEM_MALLOC_HUGE_FIRST);          \
      NPU_CHECK(aclnn_ret);                                             \
      CHECK_MEM_ALLOC(workspace_addr);                                  \
      workspace_deleter.set_workspace(workspace_addr);                  \
    }                                                                   \
    auto api_ret =                                                      \
        aclnn_api(workspace_addr, workspace_size, executor, stream);    \
    NPU_CHECK(api_ret);                                                 \
  } while (false)

}  // namespace hkv
}  // namespace npu
