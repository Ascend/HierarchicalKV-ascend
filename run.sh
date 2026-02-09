#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=h,r:,v:,i:,b:,p:,d:,c,t:
LONG=help,run-mode:,soc-version:,install-path:,build-type:,install-prefix:,device:,compile-only,enable-test:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
SOC_VERSION="ascend950pr_9579"
COMPILE_ONLY=0
ENABLE_TEST=1
RUN_MODE="npu"
BUILD_TYPE="Release"

show_help() {
    cat << EOF
Usage: ${0##*} [OPTIONS] [ARGUMENTS]
本脚本的功能描述：编译HKV工程，并执行testcase、benchmark以及demo。

OPTIONS（可选参数）：
    -h, --help                打印帮助信息并退出
    -r, --run-mode MODE       指定编译方式MODE，可选择CPU仿真、NPU上板。支持参数为[sim / npu]，默认值为npu。
    -v, --soc-version VERSION 指定昇腾AI处理器型号VERSION，默认值为Ascend910_9579。当前仅支持Ascend910_9579，请勿修改。
    -i, --install-path PATH   指定cann的安装路径PATH，默认值为环境变量ASCEND_INSTALL_PATH。
    -b, --build-type TYPE     指定构建类型TYPE，默认值为"Release"。
    -p, --install-prefix PATH 指定安装路径PATH，默认为当前路径下的out目录。
    -d, --device DEVICE_ID    指定芯片卡号DEVICE_ID，设置环境变量HKV_TEST_DEVICE，默认值为0。testcase、benchmark以及demo将指定芯片上执行。
    -c, --compile-only        仅编译安装HKV工程，不执行用例。
    -t, --enable-test SHIFT   指定测试开关SHIFT，支持参数为[0 / 1]，0表示不执行testcase，1表示执行testcase。
EOF
    exit 0
}

while :; do
    case "$1" in
    -h | --help)
        show_help
        ;;
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    -d | --device)
        export HKV_TEST_DEVICE="$2"
        shift 2
        ;;
    -c | --compile-only)
        COMPILE_ONLY=1
        shift
        ;;
    -t | --enable-test)
        ENABLE_TEST="$2"
        if ! [[ "$ENABLE_TEST" =~ ^[01]$ ]]; then
            echo "[ERROR]: ENABLE_TEST must be 0 or 1"
            exit 1
        fi
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1"
        exit 1
        ;;
    esac
done

RUN_MODE_LIST="sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "[ERROR]: RUN_MODE error, This sample only support specify sim or npu!"
    exit -1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

echo "ASCEND_HOME_PATH [$ASCEND_HOME_PATH]"
echo "_ASCEND_INSTALL_PATH [$_ASCEND_INSTALL_PATH]"

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "[INFO]: Current compile soc version is ${SOC_VERSION}"
source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
if [ "${RUN_MODE}" = "sim" ]; then
    # in case of running op in simulator, use stub .so instead
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e
rm -rf build ${INSTALL_PREFIX}
mkdir -p build
cmake -B build \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH} \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_TEST=${ENABLE_TEST}
cmake --build build -j 16 --verbose
cmake --install build

if [ $COMPILE_ONLY -eq 1 ]; then
    exit 0
fi

export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

echo "======================ctest======================"
if [ $ENABLE_TEST -eq 1 ]; then
    export LD_LIBRARY_PATH=$(pwd)/build/lib:$LD_LIBRARY_PATH
    pushd build && ctest -V
    popd
fi

(
    ${INSTALL_PREFIX}/bin/hkv_benchmark > hkv_benchmark_run.log
    ${INSTALL_PREFIX}/bin/hkv_demo > hkv_demo_run.log
)
# tidy folder by delete log files
if [ "${RUN_MODE}" = "sim" ]; then
    rm -f *.log *.dump *.vcd *.toml *_log
fi
