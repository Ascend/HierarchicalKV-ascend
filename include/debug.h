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

#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include "acl/acl.h"

namespace npu {
namespace hkv {

class NpuException : public std::runtime_error {
 public:
  NpuException(const std::string& what) : runtime_error(what) {}
};

inline void npu_check_(aclError val, const char* file, int line) {
  if (val != ACL_SUCCESS) {
    std::ostringstream os;
    os << file << ':' << line << ": aclError " << (val)
       << " aclGetRecentErrMsg: " << aclGetRecentErrMsg() << std::endl;
    throw NpuException(os.str());
  }
}

#ifdef NPU_CHECK
#error Unexpected redfinition of NPU_CHECK! Something is wrong.
#endif

#define NPU_CHECK(val)                               \
  do {                                               \
    npu::hkv::npu_check_((val), __FILE__, __LINE__); \
  } while (0)

class HkvException : public std::runtime_error {
 public:
  HkvException(const std::string& what) : runtime_error(what) {}
};

template <class Msg>
inline void hkv_check_(bool cond, const Msg& msg, const char* file, int line) {
  if (!cond) {
    std::ostringstream os;
    os << file << ':' << line << ": HierarchicalKV error " << msg;
    throw HkvException(os.str());
  }
}

#ifdef HKV_CHECK
#error Unexpected redfinition of HKV_CHECK! Something is wrong.
#endif

#define HKV_CHECK(cond, msg)                                 \
  do {                                                       \
    npu::hkv::hkv_check_((cond), (msg), __FILE__, __LINE__); \
  } while (0)

inline void log_unpack(std::queue<std::string>& fmt, std::stringstream& ss) {
  while (!fmt.empty()) {
    ss << fmt.front();
    fmt.pop();
  }
  return;
}

inline void __npuCheckError(const char* file, const int line) {
#ifdef NPU_ERROR_CHECK
  char* err = aclGetRecentErrMsg();
  if (nullptr != err) {
    fprintf(stderr, "aclGetRecentErrMsg() failed at %s:%i : %s\n", file, line,
            (err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  aclError val = aclrtSynchronizeDevice();
  if (ACL_SUCCESS != val) {
    fprintf(stderr, "aclGetRecentErrMsg() with sync failed at %s:%i : %s\n",
            file, line, aclGetRecentErrMsg());
    exit(-1);
  }
#endif
  return;
}
#define NpuCheckError() npu::hkv::__npuCheckError(__FILE__, __LINE__)

template <typename head, typename... tail>
inline void log_unpack(std::queue<std::string>& fmt, std::stringstream& ss,
                       head& h, tail&&... tails) {
  if (!fmt.empty()) {  // LCOV_EXCL_BR_LINE
    ss << fmt.front();
    fmt.pop();
  }
  ss << h;
  log_unpack(fmt, ss, tails...);
};

template <typename... Args>
inline std::string log_format(const char* fmt, Args&&... args) {
  static constexpr std::string_view DELIM = "{}";
  static constexpr size_t DELIM_LEN = DELIM.length();
  std::stringstream ss;
  std::queue<std::string> formats;
  std::string fmt_str(fmt);
  for (size_t pos = fmt_str.find_first_of(DELIM); pos != std::string::npos;
       pos = fmt_str.find_first_of(DELIM)) {
    std::string x = fmt_str.substr(0, pos);
    formats.push(x);
    fmt_str = fmt_str.substr(pos + DELIM_LEN);
  }
  formats.push(fmt_str);
  log_unpack(formats, ss, args...);
  return ss.str();
}

}  // namespace hkv
}  // namespace npu
