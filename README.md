## 概述
本目录参考https://github.com/NVIDIA-Merlin/HierarchicalKV.git，实现了昇腾A5架构下HKV HashTable的移植样例。

## 目录结构介绍
```
├── hkv                       // HKV HashTable根目录
│   ├── benchmark             // 算子性能评估工程
│   ├── cmake                 // 编译工程文件
│   ├── hkv_hashtable         // HKV算子kernel实现
│   ├── include               // HKV HashTable Host类实现
│   │   ├── hkv_hashtable.h   // HKV HashTable对外主接口头文件
│   ├── tests                 // HKV算子测试工程和测试代码
│   ├── CMakeLists.txt        // 编译工程文件
│   ├── main.cpp              // HKV应用最小demo
│   └── run.sh                // 编译运行算子的脚本
```

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/HierarchicalKV-ascend
    ```

  - 配置环境变量

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

  - 配置googletest
    注意：此安装方式仅用作本地简单验证时使用，未考虑网络安全等问题，谨慎使用！
    参考示例如下，请修改HKV_HOME为实际hkv目录路径。
    ```bash
    export HKV_HOME=/home/Ascend/HierarchicalKV-ascend
    wget https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz --no-check-certificate &&
    tar xf release-1.11.0.tar.gz && cd googletest-release-1.11.0 &&
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" -DCMAKE_INSTALL_PREFIX=$HKV_HOME/3rdparty/googletest -DCMAKE_INSTALL_LIBDIR=lib . &&
    make -j && make install &&
    cd .. && rm -rf release-1.11.0.tar.gz googletest-release-1.11.0
    ```

  - 编译与样例执行

    ```bash
    bash run.sh -d [DEVICE_ID]
    ```
    - DEVICE_ID：样例执行的npu的卡号，默认值为0。
    - **其他参数说明请使用-h打印帮助信息。**

    示例：
    ```bash
    bash run.sh -d 3
    ```