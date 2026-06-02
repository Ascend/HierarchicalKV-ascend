#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import subprocess
import sys
import os
import re
import shutil

try:
    import yaml
    from packaging.version import Version
except ImportError as e:
    print(f"[FAIL] Missing python module: {e.name}.")
    sys.exit(1)


def run(cmd_args):
    """Run a command safely without shell. cmd_args is a list, e.g. ['uname', '-m']."""
    p = subprocess.run(cmd_args, capture_output=True, text=True, check=False)
    return p.stdout.strip(), p.returncode


def check_version(actual, constraint):
    """Check if actual version satisfies the constraint like '>=3.0', '>2.5', '==1.0'."""
    m = re.match(r'(>=|>|==)(.+)', constraint)
    if not m:
        return False
    op, req_ver = m.groups()

    def _parse(v):
        try:
            return Version(v)
        except Exception:
            return tuple(map(int, re.findall(r'\d+', v)[:2]))

    actual_v, req_v = _parse(actual), _parse(req_ver)
    ops = {'>=': lambda a, b: a >= b, '>': lambda a, b: a > b, '==': lambda a, b: a == b}
    return ops.get(op, lambda a, b: False)(actual_v, req_v)


def get_arch():
    out, _ = run(["uname", "-m"])
    arch_map = {"x86_64": "x86_64-linux", "aarch64": "aarch64-linux"}
    return arch_map.get(out, out)


def check_npu_supported(npu_name):
    """Check if NPU chip is supported by scanning platform_config/*.ini files."""
    arch = get_arch()
    platform_dir = f"/usr/local/Ascend/ascend-toolkit/latest/{arch}/data/platform_config"
    if not os.path.isdir(platform_dir):
        return False
    try:
        for filename in os.listdir(platform_dir):
            if filename.endswith(".ini") and npu_name.lower() in filename.lower():
                return True
    except OSError:
        return False
    return False


def get_cann_version():
    arch = get_arch()
    paths = [
        f"/usr/local/Ascend/ascend-toolkit/latest/{arch}/ascend_toolkit_install.info",
        "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info",
        "/usr/local/Ascend/latest/ascend-toolkit/version.cfg",
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("version="):
                    return line.split("=", 1)[1].strip()
    return None


def _get_compiler_version(name):
    out, rc = run([name, "-dumpfullversion"])
    return out if rc == 0 else None


def _get_cmake_version():
    out, rc = run(["cmake", "--version"])
    if rc != 0:
        return None
    match = re.search(r'\d+\.\d+\.\d+', out.splitlines()[0])
    return match.group(0) if match else None


_COMMON_VERSION_HANDLERS = {
    "gcc": _get_compiler_version,
    "g++": _get_compiler_version,
    "cmake": lambda _: _get_cmake_version(),
}


def get_common_version(name):
    handler = _COMMON_VERSION_HANDLERS.get(name)
    return handler(name) if handler else None


def _check_item(name, constraint, actual):
    """Check one dependency. Returns True if pass. Prints [PASS]/[FAIL]."""
    if actual is None:
        print(f"[FAIL] {name} not found")
        return False
    ok = check_version(actual, constraint)
    tag = "PASS" if ok else "FAIL"
    extra = "" if ok else f" not satisfy {constraint}"
    print(f"[{tag}] {name} {actual}{extra}")
    return ok


def main():
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    yaml_path = os.path.join(root_dir, "dependencies.yaml")
    if not os.path.exists(yaml_path):
        print(f"[FAIL] {yaml_path} not found")
        sys.exit(1)

    with open(yaml_path, encoding="utf-8") as f:
        deps = yaml.safe_load(f)

    print("========== HKV Dependency Check ==========")
    fail = 0

    # common
    for item in deps.get("common", []):
        if not _check_item(item["name"], item["version"], get_common_version(item["name"])):
            fail = 1

    # ascend_stack: only check cann, skip hdk
    for item in deps.get("ascend_stack", []):
        name = item["name"]
        if name == "cann":
            if not _check_item(name, item["version"], get_cann_version()):
                fail = 1
        elif name == "hdk":
            print(f"[SKIP] {name} version check")

    # supported_npu
    for npu_name in deps.get("supported_npu", []):
        if check_npu_supported(npu_name):
            print(f"[PASS] supported npu: {npu_name}")
        else:
            print(f"[FAIL] unsupported npu: {npu_name}")
            fail = 1

    if fail:
        print("Dependency check failed, abort")
        sys.exit(1)

    # build and install HKV from source
    print("========== Building HKV ==========")
    # set default device_id to 0
    bash = shutil.which("bash") or "/bin/bash"
    rc = subprocess.run([bash, os.path.join(root_dir, "run.sh"), "-c", "-d", "0"], cwd=root_dir, check=False).returncode
    if rc != 0:
        print("[FAIL] HKV build failed")
        sys.exit(1)

    print("HKV installed")


if __name__ == "__main__":
    main()
