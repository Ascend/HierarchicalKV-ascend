# [HierarchicalKV-ascend](https://gitcode.com/Ascend/HierarchicalKV-ascend)

<div align="center">
 	  	 
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/HierarchicalKV-ascend)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/HierarchicalKV-ascend)
 	  	 
</div>

## 前言

本项目参考[NVIDIA HierarchicalKV(Beta)](https://github.com/NVIDIA-Merlin/HierarchicalKV.git)，实现了Ascend950系列架构下HKV HashTable的移植样例。

## 概述

HierarchicalKV-ascend（下称HKV）是一个基于昇腾平台，面向推荐系统的高性能key-value存储加速库。
在推荐系统中，HKV提供了大容量、高性能的动态Embedding表的增删改查能力。

## 关键特性

* 支持动态扩容
* 支持可定制的淘汰策略
* keys和values存储分离，keys仅存储于HBM
* 支持使用HOST DDR存储values

## API说明

核心类、结构体如下（命名空间均为npu::hkv::）
- `struct EvictStrategy` : 淘汰策略
- `class HashTableBase` : HashTable接口类
- `class HashTable` : HashTable实现类

核心功能均由`HashTable`各接口调用算子完成。

### API矩阵

当前已支持的接口列表及限制说明，请参考：[原生API支持度](doc/原生API支持度.md)

接口参数说明与约束请参考[HierarchicalKV API](https://nvidia-merlin.github.io/HierarchicalKV/master/api/index.html)

### 淘汰策略

引入分数（`score`）来定义每个key的重要性，分数越大，key越重要，被淘汰（`evict`）的可能性就越小。只有当一个桶（`bucket`）满时才会发生淘汰操作。`score_type`必须是`uint64_t`类型。更多详情请参考`class EvictStrategy`。
| 策略名称       | `score`意义                                                                                                     |
| :------------- | :-------------------------------------------------------------------------------------------------------------- |
| __Lru__        | 设备时钟（纳秒级），与主机时钟略有差异                                                                          |
| __Lfu__        | 调用`插值类的API`，通过`scores`参数指定的频率分数，累加到对应key的频率分数上                                    |
| __EpochLru__   | 高32位是通过`set_global_epoch`设置的全局epoch，<br>低32位等于`(device_clock >> 20) & 0xffffffff`，精度约为1毫秒 |
| __EpochLfu__   | 高32位是通过`set_global_epoch`设置的全局epoch，<br>低32位是频率分数，<br>达到最大值`0xffffffff`后频率将保持恒定 |
| __Customized__ | 由调用者调用`插值类的API`，通过`scores`传入的分数                                                               |

* __注意__：
  - 插值类API：`insert_or_assign`, `insert_and_evict`, `find_or_insert`, `accum_or_assign`, `assign_scores`以及`find_or_insert`等等。

### 如何使用

```cpp
#include "hkv_hashtable.h"

using TableOptions = npu::hkv::HashTableOptions;
using EvictStrategy = npu::hkv::EvictStrategy;

int main(int args, char *argv[])
{
  // 0. 初始化环境
  aclInit(nullptr);
  NPU_CHECK(aclrtSetDevice(0));

  using K = uint64_t;
  using V = float;
  using S = uint64_t;

  // 1. 指定淘汰策略，并创建表对象
  using HKVTable = npu::hkv::HashTable<K, V, S, EvictStrategy::kLru>;
  std::unique_ptr<HKVTable> table = std::make_unique<HKVTable>();

  // 2. 定义配置选项
  TableOptions options;
  options.init_capacity = 16 * 1024 * 1024;
  options.max_capacity = options.init_capacity;
  options.dim = 16;
  options.max_hbm_for_vectors = npu::hkv::GB(16);

  // 3. 调用init接口进行初始化
  table->init(options);

  // 4. 使用table进行相关操作
  // 5. 结束
  return 0;
}
```

### 使用约束

- `key_type`必须为`uint64_t`或`int64_t`
- `score_type`必须为`uint64_t`


## 目录结构介绍
```
├── 3rdparty           // 第三方依赖库
├── benchmark          // 算子性能评估工程
├── doc                // 文档目录
├── include            // 头文件目录
├── tests              // 单元测试与集成测试工程
├── CMakeLists.txt     // 顶层CMake构建文件
├── main.cpp           // 最小应用demo入口
└── run.sh             // 编译运行脚本
```

## 编译构建

### 构建产物

构建产物包含benchmark、demo和单元测试的可执行文件。

### 构建依赖
  - [CANN开发套件包](https://hiascend.com/document/redirect/CannCommunityInstSoftware)

  - [googletest](https://gitcode.com/gh_mirrors/googl/googletest)

### 快速构建

#### 下载代码
```bash
git clone --recurse-submodules https://gitcode.com/Ascend/HierarchicalKV-ascend
cd HierarchicalKV-ascend
```

#### 依赖配置
  - 配置CANN环境变量

    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```

#### 编译运行
```bash
bash run.sh -d [DEVICE_ID]
```
- DEVICE_ID：样例执行的npu的卡号，默认值为0。
- **其他参数说明请使用`-h`获取帮助信息。**

示例：
```bash
bash run.sh -d 3
```
