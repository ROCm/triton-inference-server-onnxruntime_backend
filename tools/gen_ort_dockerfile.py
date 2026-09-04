#!/usr/bin/env python3
# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import os
import platform
import re

FLAGS = None

ORT_TO_TRTPARSER_VERSION_MAP = {
    "1.9.0": (
        "8.2",  # TensorRT version
        "release/8.2-GA",  # ONNX-Tensorrt parser version
    ),
    "1.10.0": (
        "8.2",  # TensorRT version
        "release/8.2-GA",  # ONNX-Tensorrt parser version
    ),
}

OPENVINO_VERSION_MAP = {
    "2024.0.0": (
        "2024.0",  # OpenVINO short version
        "2024.0.0.14509.34caeefd078",  # OpenVINO version with build number
    ),
    "2024.1.0": (
        "2024.1",  # OpenVINO short version
        "2024.1.0.15008.f4afc983258",  # OpenVINO version with build number
    ),
    "2024.4.0": (
        "2024.4",  # OpenVINO short version
        "2024.4.0.16579.c3152d32c9c",  # OpenVINO version with build number
    ),
    "2024.5.0": (
        "2024.5",  # OpenVINO short version
        "2024.5.0.17288.7975fa5da0c",  # OpenVINO version with build number
    ),
    "2025.0.0": (
        "2025.0",  # OpenVINO short version
        "2025.0.0.17942.1f68be9f594",  # OpenVINO version with build number
    ),
    "2025.1.0": (
        "2025.1",  # OpenVINO short version
        "2025.1.0.18503.6fec06580ab",  # OpenVINO version with build number
    ),
    "2025.2.0": (
        "2025.2",  # OpenVINO short version
        "2025.2.0.19140.c01cd93e24d",  # OpenVINO version with build number
    ),
    "2025.3.0": (
        "2025.3",  # OpenVINO short version
        "2025.3.0.19807.44526285f24",  # OpenVINO version with build number
    ),
    "2025.4.0": (
        "2025.4",  # OpenVINO short version
        "2025.4.0.20398.8fdad55727d",  # OpenVINO version with build number
    ),
}


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


def dockerfile_common():
    df = """
ARG BASE_IMAGE={}
ARG ONNXRUNTIME_VERSION={}
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BUILD_CONFIG={}
""".format(
        FLAGS.triton_container, FLAGS.ort_version, FLAGS.ort_build_config
    )

    if FLAGS.ort_openvino is not None:
        df += """
ARG ONNXRUNTIME_OPENVINO_VERSION={}
""".format(
            FLAGS.ort_openvino
        )

    df += """
FROM ${BASE_IMAGE}
WORKDIR /workspace
"""
    return df


def dockerfile_for_linux(output_file):
    df = dockerfile_common()
    df += """
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# The Onnx Runtime dockerfile is the collection of steps in
# https://github.com/microsoft/onnxruntime/tree/master/dockerfiles

"""
    # Consider moving rhel logic to its own function e.g., dockerfile_for_rhel
    # if the changes become more substantial.
    if target_platform() == "rhel":
        df += """
# The manylinux container defaults to Python 3.7, but some feature installation
# requires a higher version.
ARG PYVER=3.12
ENV PYTHONPATH=/opt/python/v
RUN ln -sf /opt/python/cp${PYVER/./}* ${PYTHONPATH}

ENV PYBIN=${PYTHONPATH}/bin
ENV PYTHON_BIN_PATH=${PYBIN}/python${PYVER} \
    PATH=${PYBIN}:${PATH}

RUN dnf install -y \\
        ca-certificates \\
        curl \\
        git \\
        gnupg \\
        gnupg1 \\
        openssl-devel \\
        python3-pip \\
        wget \\
        zip

RUN pipx install cmake==3.31.10 --force

RUN pip3 install patchelf==0.17.2 numpy>=2.0.0
"""
    else:
        if os.getenv("CCACHE_REMOTE_ONLY") and os.getenv("CCACHE_REMOTE_STORAGE"):
            df += """
ENV CCACHE_REMOTE_ONLY="true" \\
    CCACHE_REMOTE_STORAGE="{}" \\
    CMAKE_CXX_COMPILER_LAUNCHER="ccache" \\
    CMAKE_C_COMPILER_LAUNCHER="ccache" \\
    CMAKE_CUDA_COMPILER_LAUNCHER="ccache" \\
    VERBOSE=1

RUN apt-get update \\
      && apt-get install -y --no-install-recommends ccache && ccache -p \\
      && rm -rf /var/lib/apt/lists/*
""".format(
                os.getenv("CCACHE_REMOTE_STORAGE")
            )

        df += """

RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential \\
        ca-certificates \\
        curl \\
        git \\
        gnupg \\
        gnupg1 \\
        libcurl4-openssl-dev \\
        libssl-dev \\
        python3-dev \\
        python3-pip \\
        software-properties-common \\
        wget \\
        zip

RUN pip3 install \\
       cmake==3.31.10 \\
       numpy \\
       packaging \\
       patchelf==0.17.2 \\
       wheel>=0.35.1

ENV VERBOSE=1
"""

    # ROCm: install build tools; MIGraphX and ONNX Runtime are built from source below
    if FLAGS.enable_rocm:
        df += """
# ROCm: install build tools (MIGraphX and ONNX Runtime built from source in this image)
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        sudo git apt-utils bash build-essential curl \\
        python3-dev python3-pip aria2 libnuma-dev pkg-config ccache \\
    && rm -rf /var/lib/apt/lists/*
"""

    if FLAGS.ort_openvino is not None:
        df += """
# Install OpenVINO
ARG ONNXRUNTIME_OPENVINO_VERSION
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}
"""
        df += """
ARG OPENVINO_SHORT_VERSION={}
ARG OPENVINO_VERSION_WITH_BUILD_NUMBER={}
""".format(
            OPENVINO_VERSION_MAP[FLAGS.ort_openvino][0],
            OPENVINO_VERSION_MAP[FLAGS.ort_openvino][1],
        )

        # Openvino changed the filename of the toolkit in 2025.0.0 so we need to detect this for
        # the release we want to install
        openvino_folder_name = "UNKNOWN_FOLDER_NAME"
        openvino_toolkit_filename = "UNKNOWN_FILENAME"
        if OPENVINO_VERSION_MAP[FLAGS.ort_openvino][0].split(".")[0] >= "2025":
            openvino_folder_name = (
                "openvino_toolkit_ubuntu24_${OPENVINO_VERSION_WITH_BUILD_NUMBER}_x86_64"
            )
            openvino_toolkit_filename = openvino_folder_name + ".tgz"
        else:
            openvino_folder_name = "l_openvino_toolkit_ubuntu24_${OPENVINO_VERSION_WITH_BUILD_NUMBER}_x86_64"
            openvino_toolkit_filename = openvino_folder_name + ".tgz"

        df += """
# Step 1: Download and install core components
# Ref: https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html#step-1-download-and-install-the-openvino-core-components
RUN curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/${{OPENVINO_SHORT_VERSION}}/linux/{} --output openvino_${{ONNXRUNTIME_OPENVINO_VERSION}}.tgz && \
    tar -xf openvino_${{ONNXRUNTIME_OPENVINO_VERSION}}.tgz && \
    mkdir -p ${{INTEL_OPENVINO_DIR}} && \
    mv {}/* ${{INTEL_OPENVINO_DIR}} && \
    rm openvino_${{ONNXRUNTIME_OPENVINO_VERSION}}.tgz && \
    (cd ${{INTEL_OPENVINO_DIR}}/install_dependencies && \
        ./install_openvino_dependencies.sh -y) && \
    ln -s ${{INTEL_OPENVINO_DIR}} ${{INTEL_OPENVINO_DIR}}/../openvino_`echo ${{ONNXRUNTIME_OPENVINO_VERSION}} | awk '{{print substr($0,0,4)}}'`

# Step 2: Configure the environment
# Ref: https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html#step-2-configure-the-environment
ENV OpenVINO_DIR=$INTEL_OPENVINO_DIR/runtime/cmake
ENV LD_LIBRARY_PATH=$INTEL_OPENVINO_DIR/runtime/lib/intel64:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=$INTEL_OPENVINO_DIR/runtime/lib/intel64/pkgconfig
ENV PYTHONPATH=$INTEL_OPENVINO_DIR/python/python3.12:$INTEL_OPENVINO_DIR/python/python3:$PYTHONPATH
""".format(
            openvino_toolkit_filename, openvino_folder_name
        )

    # ROCm: provision MIGraphX (either build the specified branch from source
    # with a prebuilt rocMLIR, or pull prebuilt MIGraphX packages), then fetch
    # the prebuilt ONNX Runtime core and build the out-of-tree plugin EP.
    if FLAGS.enable_rocm:
        if FLAGS.migraphx_build_mode == "package":
            df += """
#
# Provision MIGraphX from prebuilt packages (no MIGraphX/rocMLIR source build).
#
# Pulls amdrocm-migraphx (runtime libs) and amdrocm-migraphx-dev (headers +
# migraphxConfig.cmake). The prebuilt library already has rocMLIR statically
# linked, so there is NO LLVM/MLIR compile and NO MIGraphX source build -- only
# a package download + install. MIGRAPHX_PACKAGE_VERSION must match the base
# image ROCm train (e.g. 2.17.0 => rocm10.0.0) and Debian release (debian12).
#
ARG MIGRAPHX_PACKAGE_URL={}
ARG MIGRAPHX_PACKAGE_VERSION={}
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_BUILD_CONFIG

RUN (command -v curl >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends curl ca-certificates)) && \\
    cd /tmp && \\
    MIGRAPHX_PACKAGE_VERSION_URL="$(printf '%s' "${{MIGRAPHX_PACKAGE_VERSION}}" | sed 's/+/%2B/g')" && \\
    curl -fSL -o amdrocm-migraphx_${{MIGRAPHX_PACKAGE_VERSION}}_amd64.deb ${{MIGRAPHX_PACKAGE_URL}}/amdrocm-migraphx_${{MIGRAPHX_PACKAGE_VERSION_URL}}_amd64.deb && \\
    curl -fSL -o amdrocm-migraphx-dev_${{MIGRAPHX_PACKAGE_VERSION}}_amd64.deb ${{MIGRAPHX_PACKAGE_URL}}/amdrocm-migraphx-dev_${{MIGRAPHX_PACKAGE_VERSION_URL}}_amd64.deb && \\
    (dpkg -i --force-depends /tmp/amdrocm-migraphx_${{MIGRAPHX_PACKAGE_VERSION}}_amd64.deb /tmp/amdrocm-migraphx-dev_${{MIGRAPHX_PACKAGE_VERSION}}_amd64.deb || (apt-get update && apt-get install -f -y)) && \\
    rm -f /tmp/amdrocm-migraphx*.deb && \\
    find /opt/rocm -iname 'migraphx*config*.cmake' -o -iname 'migraphxConfig.cmake' | tee /tmp/migraphx_cmake_files.txt && \\
    if [ ! -s /tmp/migraphx_cmake_files.txt ]; then \\
        echo "ERROR: migraphx CMake package config not found under /opt/rocm after installing prebuilt packages; find_package(migraphx) will fail downstream. Verify --migraphx-package-version matches the base ROCm train and Debian release (e.g. 2.17.0+rocm10.0.0 for debian12)."; \\
        exit 1; \\
    fi
""".format(
                FLAGS.migraphx_package_url,
                FLAGS.migraphx_package_version,
            )
        else:
            df += """
#
# Build rocMLIR (librockCompiler) ONCE from a pinned commit into /opt/rocmlir.
#
# This is the expensive LLVM/MLIR compile. MIGraphX consumes rocMLIR as the
# static rocMLIR::rockCompiler library (find_package(rocMLIR) in
# src/targets/gpu/CMakeLists.txt), so building it here as its own Docker layer --
# keyed only by ROCMLIR_REPO/ROCMLIR_COMMIT -- decouples the LLVM/MLIR build from
# the (frequently-changing) MIGraphX and plugin-EP source: changing
# MIGRAPHX_BRANCH or the EP no longer triggers an LLVM rebuild. ROCMLIR_COMMIT
# MUST match the ROCm/rocMLIR@<sha> pin in MIGraphX's requirements.txt for ABI
# compatibility; bump it to force a rocMLIR rebuild.
#
ARG ROCMLIR_REPO={}
ARG ROCMLIR_COMMIT={}
RUN pip3 install --no-cache-dir ninja && \\
    ROCM_CLANGXX=""; for c in /opt/rocm/llvm/bin/clang++ /opt/rocm/lib/llvm/bin/clang++; do if [ -x "$c" ]; then ROCM_CLANGXX="$c"; break; fi; done && \\
    CC_ARGS=""; if [ -n "$ROCM_CLANGXX" ]; then CC_ARGS="-DCMAKE_CXX_COMPILER=$ROCM_CLANGXX -DCMAKE_C_COMPILER=${{ROCM_CLANGXX%++}}"; fi && \\
    echo "Building rocMLIR ${{ROCMLIR_COMMIT}} (C++ compiler: ${{ROCM_CLANGXX:-<system default>}})" && \\
    git init rocmlir_src && cd rocmlir_src && \\
    git remote add origin ${{ROCMLIR_REPO}} && \\
    git fetch --depth 1 origin ${{ROCMLIR_COMMIT}} && \\
    git checkout --detach FETCH_HEAD && \\
    mkdir -p build && cd build && \\
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FAT_LIBROCKCOMPILER=On -DLLVM_INCLUDE_TESTS=Off $CC_ARGS 2>&1 | tee /tmp/rocmlir_cmake.log && \\
    ninja 2>&1 | tee /tmp/rocmlir_build.log && \\
    cmake --install . --prefix /opt/rocmlir && \\
    find /opt/rocmlir -name 'rocMLIRConfig.cmake' -o -name 'rocmlir-config.cmake' | tee /tmp/rocmlir_cmake_files.txt && \\
    if [ ! -s /tmp/rocmlir_cmake_files.txt ]; then \\
        echo "ERROR: rocMLIR CMake package config not found under /opt/rocmlir after install; find_package(rocMLIR) will fail in the MIGraphX build."; \\
        exit 1; \\
    fi && \\
    cd / && rm -rf /workspace/rocmlir_src

#
# Build MIGraphX from source, reusing the prebuilt rocMLIR above.
#
# The ROCm/rocMLIR@<sha> line is stripped from requirements.txt so rbuild does
# NOT rebuild LLVM/MLIR; MIGraphX's find_package(rocMLIR) is instead pointed at
# the prebuilt /opt/rocmlir via rocMLIR_DIR (+ CMAKE_PREFIX_PATH).
#
ARG MIGRAPHX_REPO={}
ARG MIGRAPHX_BRANCH={}
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_BUILD_CONFIG

RUN pip3 install --no-cache-dir wheel build && \\
    git clone ${{MIGRAPHX_REPO}} --recursive -b ${{MIGRAPHX_BRANCH}} migraphx_src && \\
    cd migraphx_src && \\
    pip3 install --no-cache-dir https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz && \\
    EXPECTED_MLIR=$(grep -oiE 'rocMLIR@[0-9a-f]+' requirements.txt | head -1 | cut -d@ -f2) && \\
    if [ -n "$EXPECTED_MLIR" ] && [ "$EXPECTED_MLIR" != "${{ROCMLIR_COMMIT}}" ]; then \\
        echo "WARNING: prebuilt rocMLIR ${{ROCMLIR_COMMIT}} != MIGraphX requirements.txt pin $EXPECTED_MLIR; rebuild rocMLIR at the matching commit (set --rocmlir-commit) to avoid an ABI mismatch."; \\
    fi && \\
    sed -i '/rocMLIR/d' requirements.txt && \\
    ROCMLIR_CFG=$(find /opt/rocmlir -name 'rocMLIRConfig.cmake' -o -name 'rocmlir-config.cmake' 2>/dev/null | head -1) && \\
    if [ -z "$ROCMLIR_CFG" ]; then echo "ERROR: prebuilt rocMLIR not found under /opt/rocmlir; the rocMLIR build stage must run first."; exit 1; fi && \\
    export CMAKE_PREFIX_PATH="/opt/rocmlir:${{CMAKE_PREFIX_PATH:-}}" && \\
    rbuild build -d depend -B build -DMIGRAPHX_ENABLE_PYTHON=OFF -DGPU_TARGETS=gfx942 \\
        -DMIGRAPHX_PACKAGE_BACKEND=default -DrocMLIR_DIR="$(dirname $ROCMLIR_CFG)" 2>&1 | tee migraphx_build.log && \\
    cd build && \\
    make -j$(nproc) package && dpkg -i --force-depends *.deb && \\
    find /opt/rocm -iname 'migraphx*config*.cmake' -o -iname 'migraphxConfig.cmake' | tee /tmp/migraphx_cmake_files.txt && \\
    if [ ! -s /tmp/migraphx_cmake_files.txt ]; then \\
        echo "ERROR: migraphx CMake package config not found under /opt/rocm after dpkg install; find_package(migraphx) will fail downstream."; \\
        exit 1; \\
    fi
""".format(
                FLAGS.rocmlir_repo,
                FLAGS.rocmlir_commit,
                FLAGS.migraphx_repo,
                FLAGS.migraphx_branch,
            )

        # Shared by both MIGraphX provisioning modes: MLIR runtime env vars, the
        # prebuilt ONNX Runtime core, and the out-of-tree MIGraphX plugin EP.
        df += """
ENV MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention,dot
ENV MIGRAPHX_ENABLE_MLIR_GEG_FUSION=1
ENV MIGRAPHX_ENABLE_REWRITE_DOT=1

#
# Fetch prebuilt ONNX Runtime core (EP-agnostic) from the upstream GitHub
# release instead of building it from source. ONNX Runtime core does not need
# ROCm: all MIGraphX support is provided out-of-tree by the plugin EP built
# below, so the generic linux-x64 package is sufficient and avoids the long,
# network-heavy ONNX Runtime source build.
#
# The upstream release tarball ships include/ + lib/ only (no CMake package
# config), so we synthesize a minimal onnxruntimeConfig.cmake that exposes the
# onnxruntime::onnxruntime imported target expected by the plugin's
# find_package(onnxruntime) (src/CMakeLists.txt).
#
ARG ONNXRUNTIME_DIST=/opt/onnxruntime-dist
RUN (command -v curl >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends curl ca-certificates)) && \\
    mkdir -p ${{ONNXRUNTIME_DIST}} && \\
    cd /tmp && \\
    curl -fSL -o ort.tgz \\
      https://github.com/microsoft/onnxruntime/releases/download/v${{ONNXRUNTIME_VERSION}}/onnxruntime-linux-x64-${{ONNXRUNTIME_VERSION}}.tgz && \\
    tar -xzf ort.tgz --strip-components=1 -C ${{ONNXRUNTIME_DIST}} && \\
    rm ort.tgz && \\
    mkdir -p ${{ONNXRUNTIME_DIST}}/lib/cmake/onnxruntime && \\
    printf '%s\\n' \\
      'get_filename_component(_ort_root "${{CMAKE_CURRENT_LIST_DIR}}/../../.." ABSOLUTE)' \\
      'add_library(onnxruntime::onnxruntime SHARED IMPORTED)' \\
      'set_target_properties(onnxruntime::onnxruntime PROPERTIES' \\
      '  IMPORTED_LOCATION "${{_ort_root}}/lib/libonnxruntime.so"' \\
      '  INTERFACE_INCLUDE_DIRECTORIES "${{_ort_root}}/include")' \\
      "set(onnxruntime_VERSION ${{ONNXRUNTIME_VERSION}})" \\
      > ${{ONNXRUNTIME_DIST}}/lib/cmake/onnxruntime/onnxruntimeConfig.cmake && \\
    echo "ONNX Runtime ${{ONNXRUNTIME_VERSION}} prebuilt core staged at ${{ONNXRUNTIME_DIST}}"

#
# Build the out-of-tree MIGraphX plugin EP (onnxruntime-ep-amdgpu ->
# libmigraphx-ep.so), following the same steps as
# onnxruntime-ep-amdgpu/scripts/build_migraphx_ep_standalone.sh (its step 3:
# "Build onnxruntime-ep-amdgpu"). The plugin is built against the prebuilt ONNX
# Runtime core staged at ${{ONNXRUNTIME_DIST}} (--onnxrt_home) and the MIGraphX
# install under /opt/rocm (--migraphx_home, from the dpkg install above). The
# resulting libmigraphx-ep.so is registered at runtime by the Triton backend via
# RegisterExecutionProviderLibrary.
#
# Dependency setup mirrors the standalone script's step 0 (ninja,
# packaging>=24.2, cmake==4.2.3, CXXFLAGS="-D__HIP_PLATFORM_AMD__=1 -w").
# onnxruntime-ep-amdgpu/CMakeLists.txt requires CMake >= 4.2, which is newer than
# the cmake shipped in the base image, so we install a compatible cmake via pip
# for this build step only and point build.sh at it explicitly with --cmake_path,
# leaving the image's system cmake untouched for every other build stage.
#
ARG MIGRAPHX_EP_REPO={}
ARG MIGRAPHX_EP_BRANCH={}
# Cache-bust token for just this EP stage. When Triton passes a new value
# (--build-arg MIGRAPHX_EP_CACHE_BUST=...), Docker rebuilds only the EP below and
# reuses the cached MIGraphX and prebuilt ONNX Runtime core layers above.
ARG MIGRAPHX_EP_CACHE_BUST
RUN echo "MIGraphX plugin EP cache-bust token: ${{MIGRAPHX_EP_CACHE_BUST}}" && \\
    rm -rf onnxruntime-ep-amdgpu && \\
    git clone ${{MIGRAPHX_EP_REPO}} --recursive -b ${{MIGRAPHX_EP_BRANCH}} onnxruntime-ep-amdgpu && \\
    cd onnxruntime-ep-amdgpu && \\
    git config --global --add safe.directory "*" && \\
    pip3 install --no-cache-dir --upgrade ninja "packaging>=24.2" cmake==4.2.3 && \\
    MGX_EP_CMAKE_BIN=$(python3 -c "import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin'))") && \\
    export PATH="$MGX_EP_CMAKE_BIN:$PATH" && \\
    export CXXFLAGS="-D__HIP_PLATFORM_AMD__=1 -w" && \\
    ./build.sh --config ${{ONNXRUNTIME_BUILD_CONFIG}} \\
        --cmake_generator Ninja \\
        --cmake_path "$MGX_EP_CMAKE_BIN/cmake" \\
        --onnxrt_home ${{ONNXRUNTIME_DIST}} \\
        --use_migraphx \\
        --migraphx_home /opt/rocm \\
        --compile_no_warning_as_error \\
        --parallel $(nproc) \\
        --build_dir build.EP.MGX \\
        --hip_path /opt/rocm 2>&1 | tee migraphx_ep_build.log; \\
    if [ ! -f build.EP.MGX/${{ONNXRUNTIME_BUILD_CONFIG}}/src/migraphx/libmigraphx-ep.so ]; then \\
        echo "ERROR: MIGraphX plugin EP build failed; libmigraphx-ep.so was not produced. See the build output / migraphx_ep_build.log above for the underlying error (e.g. an unrecognized build.sh flag, find_package(onnxruntime) not resolvable under ${{ONNXRUNTIME_DIST}}, or find_package(migraphx) not resolvable under /opt/rocm)."; \\
        exit 1; \\
    fi && \\
    echo "MIGraphX plugin EP built at /workspace/onnxruntime-ep-amdgpu/build.EP.MGX/${{ONNXRUNTIME_BUILD_CONFIG}}/src/migraphx/libmigraphx-ep.so"
""".format(
            FLAGS.migraphx_ep_repo,
            FLAGS.migraphx_ep_branch,
        )
    ## TEMPORARY: Using the tensorrt-8.0 branch until ORT 1.9 release to enable ORT backend with TRT 8.0 support.
    # For ORT versions 1.8.0 and below the behavior will remain same. For ORT version 1.8.1 we will
    # use tensorrt-8.0 branch instead of using rel-1.8.1
    # From ORT 1.9 onwards we will switch back to using rel-* branches
    elif FLAGS.ort_version == "1.8.1":
        df += """
#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
ARG ONNXRUNTIME_BUILD_CONFIG

RUN git clone -b tensorrt-8.0 --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
       """
    # Use the tensorrt-8.5ea branch to use Tensor RT 8.5a to use the built-in tensorrt parser
    elif FLAGS.ort_version == "1.12.1":
        df += """
#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
ARG ONNXRUNTIME_BUILD_CONFIG

RUN git clone -b tensorrt-8.5ea --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
       """
    else:
        df += """
#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
ARG ONNXRUNTIME_BUILD_CONFIG

RUN git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime
        """

    # Skip onnx-tensorrt tag and build for ROCm (using pre-built)
    if not FLAGS.enable_rocm:
        if FLAGS.onnx_tensorrt_tag != "":
            df += """
    RUN (cd /workspace/onnxruntime/cmake/external/onnx-tensorrt && git fetch origin {}:ortrefbranch && git checkout ortrefbranch)
    """.format(
                FLAGS.onnx_tensorrt_tag
            )

    ep_flags = ""
    if FLAGS.enable_gpu:
        ep_flags = "--use_cuda"
        if FLAGS.cuda_version is not None:
            ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
        if FLAGS.cuda_home is not None:
            ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
        if FLAGS.cudnn_home is not None:
            ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
        elif target_platform() == "igpu":
            ep_flags += ' --cudnn_home "/usr/include"'
        if FLAGS.ort_tensorrt:
            ep_flags += " --use_tensorrt"
            if FLAGS.ort_version >= "1.12.1":
                ep_flags += " --use_tensorrt_builtin_parser"
            if FLAGS.tensorrt_home is not None:
                ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)

    if os.name == "posix":
        if os.getuid() == 0:
            ep_flags += " --allow_running_as_root"

    if FLAGS.ort_openvino is not None:
        ep_flags += " --use_openvino CPU"

    if target_platform() == "igpu":
        ep_flags += (
            " --skip_tests --cmake_extra_defines 'onnxruntime_BUILD_UNIT_TESTS=OFF'"
        )
        if os.getenv("CUDA_ARCH_LIST") is not None:
            print(f"[INFO] Defined CUDA_ARCH_LIST: {os.getenv('CUDA_ARCH_LIST')}")
            cuda_archs = (
                os.getenv("CUDA_ARCH_LIST")
                .replace("PTX", "")
                .replace(" ", "-real;")
                .replace(".", "")
            )
            cuda_archs = re.sub(r"-real;$", "", cuda_archs)
            print(f"[INFO] Set ONNX Runtime to use CUDA architectures to: {cuda_archs}")
        else:
            cuda_archs = "87"
    else:
        if os.uname().machine != "x86_64":
            cuda_archs = "80;86;90;100;110;120;121"
        elif os.getenv("CUDA_ARCH_LIST") is not None:
            print(f"[INFO] Defined CUDA_ARCH_LIST: {os.getenv('CUDA_ARCH_LIST')}")
            cuda_archs = (
                os.getenv("CUDA_ARCH_LIST")
                .replace("PTX", "")
                .replace(" ", "-real;")
                .replace(".", "")
            )
            cuda_archs = re.sub(r"-real;$", "", cuda_archs)
            print(f"[INFO] Set ONNX Runtime to use CUDA architectures to: {cuda_archs}")
        else:
            cuda_archs = "75;80;86;90;100;120"

    # Skip build.sh for ROCm (using pre-built)
    if not FLAGS.enable_rocm:
        df += """
WORKDIR /workspace/onnxruntime
ARG COMMON_BUILD_ARGS="--config ${{ONNXRUNTIME_BUILD_CONFIG}} --parallel --skip_submodule_sync --build_shared_lib \\
    --compile_no_warning_as_error --build_dir /workspace/build --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES='{}'  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 --build_wheel"
""".format(
            cuda_archs
        )

        df += """
RUN ./build.sh ${{COMMON_BUILD_ARGS}} --update --build {}
""".format(
            ep_flags
        )

    # ROCm: Copy built ONNX Runtime artifacts to /opt/onnxruntime
    if FLAGS.enable_rocm:
        df += """
#
# Copy ONNX Runtime artifacts from build to /opt/onnxruntime
#
WORKDIR /workspace

RUN mkdir -p /opt/onnxruntime/lib /opt/onnxruntime/include

# Copy the ONNX Runtime core shared libraries from the prebuilt dist.
# Note: the built-in MIGraphX provider is no longer built; MIGraphX is provided
# by the plugin EP (libmigraphx-ep.so) staged below. The prebuilt tarball may not
# ship libonnxruntime_providers_shared.so, which is not required by the plugin EP
# path, so that copy is best-effort.
RUN cp -P /opt/onnxruntime-dist/lib/libonnxruntime.so* /opt/onnxruntime/lib/ && \\
    cd /opt/onnxruntime/lib && \\
    ORT_SO=$(basename "$(readlink -f libonnxruntime.so)") && \\
    ln -sf "$ORT_SO" libonnxruntime.so.1 && \\
    ln -sf "$ORT_SO" libonnxruntime.so && \\
    (cp /opt/onnxruntime-dist/lib/libonnxruntime_providers_shared.so /opt/onnxruntime/lib/ 2>/dev/null || \\
     echo "libonnxruntime_providers_shared.so not present in prebuilt dist (not required for plugin EP)")

# Stage the MIGraphX plugin EP shared library. The Triton ONNX Runtime backend
# registers this at runtime via RegisterExecutionProviderLibrary; the default
# lookup path is /opt/tritonserver/backends/onnxruntime/libmigraphx-ep.so, which
# is where the backend install places the contents of /opt/onnxruntime/lib.
RUN cp /workspace/onnxruntime-ep-amdgpu/build.EP.MGX/${ONNXRUNTIME_BUILD_CONFIG}/src/migraphx/libmigraphx-ep.so /opt/onnxruntime/lib/ && \\
    echo "MIGraphX plugin EP (libmigraphx-ep.so) staged to /opt/onnxruntime/lib"

# Copy MIGraphX runtime libraries (built in this image via dpkg) into /opt/onnxruntime/lib
# so they are included in final artifacts; the provider loads libmigraphx_c.so.3 at runtime.
# Search /usr, /opt/rocm, /usr/local, and /workspace (dpkg installs to /opt/rocm; build tree under /workspace).
RUN find /usr /opt/rocm /usr/local /workspace -name 'libmigraphx*.so*' 2>/dev/null | while read f; do cp -P "$f" /opt/onnxruntime/lib/; done && \\
    (ls /opt/onnxruntime/lib/libmigraphx*.so* 2>/dev/null && echo "MIGraphX runtime libs copied to /opt/onnxruntime/lib") || echo "No MIGraphX libs found under /usr, /opt/rocm, /usr/local, or /workspace"

# Copy the full ONNX Runtime header set from the prebuilt dist. ORT splits the C
# API across several headers (e.g. onnxruntime_c_api.h -> onnxruntime_error_code.h,
# onnxruntime_ep_c_api.h, onnxruntime_cxx_api.h, ...), so copy the whole include/
# tree rather than a hand-picked subset to stay robust across versions. The
# backend install excludes include/ from the final artifact, so this only affects
# the compile step, not the packaged backend size.
RUN echo "Copying ONNX Runtime headers from /opt/onnxruntime-dist/include/" && \\
    cp -a /opt/onnxruntime-dist/include/. /opt/onnxruntime/include/ && \\
    echo "${ONNXRUNTIME_VERSION}" > /opt/onnxruntime/ort_onnx_version.txt && \\
    echo "ONNX Runtime headers copied to /opt/onnxruntime/include"

# Set RPATH for all .so files
RUN cd /opt/onnxruntime/lib && \\
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\\.so*'`; do \\
        patchelf --set-rpath '$ORIGIN' $i || true; \\
    done

# Create bin and test directories
RUN mkdir -p /opt/onnxruntime/bin /opt/onnxruntime/test
"""
        # Write dockerfile and return early for ROCm
        with open(output_file, "w") as dfile:
            dfile.write(df)
        return

    df += """
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime
RUN cp /workspace/onnxruntime/LICENSE . \\
    && cat /workspace/onnxruntime/cmake/external/onnx/VERSION_NUMBER > ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
WORKDIR /opt/onnxruntime/include
RUN cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h . \\
    && cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h . \\
    && cp /workspace/onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h . \\
    && cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_ep_c_api.h .

WORKDIR /opt/onnxruntime/lib
RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_shared.so . \\
    && cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime.so .
"""
    if target_platform() == "igpu":
        df += """
RUN mkdir -p /opt/onnxruntime/bin
"""
    else:
        df += """
WORKDIR /opt/onnxruntime/bin
RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnxruntime_perf_test . \\
    && cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnx_test_runner . \\
    && chmod a+x *
"""

    if FLAGS.enable_gpu:
        df += """
RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_cuda.so \
       /opt/onnxruntime/lib
"""

    if FLAGS.ort_tensorrt:
        df += """
# TensorRT specific headers and libraries
RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_tensorrt.so \
       /opt/onnxruntime/lib
"""

    if FLAGS.ort_openvino is not None:
        df += """
# OpenVino specific headers and libraries
RUN cp -r ${INTEL_OPENVINO_DIR}/docs/licensing /opt/onnxruntime/LICENSE.openvino

RUN cp /workspace/onnxruntime/include/onnxruntime/core/providers/openvino/openvino_provider_factory.h \
       /opt/onnxruntime/include

RUN apt-get update && apt-get install -y --no-install-recommends libtbb12

RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_openvino.so \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp /usr/lib/x86_64-linux-gnu/libtbb.so.* /opt/onnxruntime/lib

RUN OV_SHORT_VERSION=`echo ${ONNXRUNTIME_OPENVINO_VERSION} | awk '{ split($0,a,"."); print substr(a[1],3) a[2] a[3] }'` && \
    (cd /opt/onnxruntime/lib && \
        chmod a-x * && \
        ln -s libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino.so && \
        ln -s libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_c.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_c.so && \
        ln -s libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_ir_frontend.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_ir_frontend.so && \
        ln -s libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_onnx_frontend.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_onnx_frontend.so)
"""
    # Linking compiled ONNX Runtime libraries to their corresponding versioned libraries
    df += """
RUN cd /opt/onnxruntime/lib \
        && ln -s libonnxruntime.so libonnxruntime.so.1 \
        && ln -s libonnxruntime.so.1 libonnxruntime.so.${ONNXRUNTIME_VERSION}
"""
    df += """
RUN cd /opt/onnxruntime/lib && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# For testing copy ONNX custom op library and model
"""
    if target_platform() == "igpu":
        df += """
RUN mkdir -p /opt/onnxruntime/test
"""
    else:
        df += """
RUN mkdir -p /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libcustom_op_library.so \
       /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/testdata/custom_op_library/custom_op_test.onnx \
       /opt/onnxruntime/test
"""

    with open(output_file, "w") as dfile:
        dfile.write(df)


def dockerfile_for_windows(output_file):
    df = dockerfile_common()

    ## TEMPORARY: Using the tensorrt-8.0 branch until ORT 1.9 release to enable ORT backend with TRT 8.0 support.
    # For ORT versions 1.8.0 and below the behavior will remain same. For ORT version 1.8.1 we will
    # use tensorrt-8.0 branch instead of using rel-1.8.1
    # From ORT 1.9 onwards we will switch back to using rel-* branches
    if FLAGS.ort_version == "1.8.1":
        df += """
SHELL ["cmd", "/S", "/C"]

#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO

RUN git clone -b tensorrt-8.0 --recursive %ONNXRUNTIME_REPO% onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
"""
    else:
        df += """
SHELL ["cmd", "/S", "/C"]

#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
RUN git clone -b rel-%ONNXRUNTIME_VERSION% --recursive %ONNXRUNTIME_REPO% onnxruntime && \
    cd onnxruntime && git submodule update --init --recursive
"""

    if FLAGS.onnx_tensorrt_tag != "":
        df += """
    RUN (cd \\workspace\\onnxruntime\\cmake\\external\\onnx-tensorrt && git fetch origin {}:ortrefbranch && git checkout ortrefbranch)
    """.format(
            FLAGS.onnx_tensorrt_tag
        )

    ep_flags = ""
    if FLAGS.enable_gpu:
        ep_flags = "--use_cuda"
        if FLAGS.cuda_version is not None:
            ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
        if FLAGS.cuda_home is not None:
            ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
        if FLAGS.cudnn_home is not None:
            ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
        if FLAGS.ort_tensorrt:
            ep_flags += " --use_tensorrt"
            if FLAGS.tensorrt_home is not None:
                ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)
    if FLAGS.ort_openvino is not None:
        ep_flags += " --use_openvino CPU"

    df += """
WORKDIR /workspace/onnxruntime
ARG VS_DEVCMD_BAT="\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
RUN powershell Set-Content 'build.bat' -value 'call %VS_DEVCMD_BAT%',(Get-Content 'build.bat')
RUN build.bat --cmake_generator "Visual Studio 17 2022" --config Release --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=75;80;86;90;100;120" --skip_submodule_sync --parallel --build_shared_lib --compile_no_warning_as_error --skip_tests --build_wheel --update --build --build_dir /workspace/build {}
""".format(
        ep_flags
    )

    df += """
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime
RUN copy \\workspace\\onnxruntime\\LICENSE \\opt\\onnxruntime
RUN copy \\workspace\\onnxruntime\\cmake\\external\\onnx\\VERSION_NUMBER \\opt\\onnxruntime\\ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
WORKDIR /opt/onnxruntime/include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_c_api.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_session_options_config_keys.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\providers\\cpu\\cpu_provider_factory.h \\opt\\onnxruntime\\include

WORKDIR /opt/onnxruntime/bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_perf_test.exe \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnx_test_runner.exe \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.lib \\opt\\onnxruntime\\lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.lib \\opt\\onnxruntime\\lib
"""

    if FLAGS.enable_gpu:
        df += """
WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.lib \\opt\\onnxruntime\\lib
WORKDIR /opt/onnxruntime/bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.dll \\opt\\onnxruntime\\bin
"""

    if FLAGS.ort_tensorrt:
        df += """
# TensorRT specific headers and libraries
WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.dll \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.lib \\opt\\onnxruntime\\lib
"""
    with open(output_file, "w") as dfile:
        dfile.write(df)


def preprocess_gpu_flags():
    if target_platform() == "windows":
        # Default to CUDA based on CUDA_PATH envvar and TensorRT in
        # C:/tensorrt
        if "CUDA_PATH" in os.environ:
            if FLAGS.cuda_home is None:
                FLAGS.cuda_home = os.environ["CUDA_PATH"]
            elif FLAGS.cuda_home != os.environ["CUDA_PATH"]:
                print("warning: --cuda-home does not match CUDA_PATH envvar")

        if FLAGS.cudnn_home is None:
            FLAGS.cudnn_home = FLAGS.cuda_home

        version = None
        m = re.match(r".*v([1-9]?[0-9]+\.[0-9]+)$", FLAGS.cuda_home)
        if m:
            version = m.group(1)

        if FLAGS.cuda_version is None:
            FLAGS.cuda_version = version
        elif FLAGS.cuda_version != version:
            print("warning: --cuda-version does not match CUDA_PATH envvar")

        if (FLAGS.cuda_home is None) or (FLAGS.cuda_version is None):
            print("error: windows build requires --cuda-version and --cuda-home")

        if FLAGS.tensorrt_home is None:
            FLAGS.tensorrt_home = "/tensorrt"
    else:
        if not FLAGS.enable_rocm:
            if "CUDNN_VERSION" in os.environ:
                if FLAGS.cudnn_home is None:
                    FLAGS.cudnn_home = "/usr"

            if FLAGS.cuda_home is None:
                FLAGS.cuda_home = "/usr/local/cuda"

            if (FLAGS.cuda_home is None) or (FLAGS.cudnn_home is None):
                print("error: linux build requires --cudnn-home and --cuda-home")

        if FLAGS.tensorrt_home is None:
            if target_platform() == "rhel":
                if platform.machine().lower() == "aarch64":
                    FLAGS.tensorrt_home = "/usr/local/cuda/targets/sbsa-linux/"
                else:
                    FLAGS.tensorrt_home = "/usr/local/cuda/targets/x86_64-linux/"
            else:
                FLAGS.tensorrt_home = "/usr/src/tensorrt"

        # ROCm defaults
        if FLAGS.enable_rocm:
            if FLAGS.rocm_home is None:
                FLAGS.rocm_home = "/opt/rocm"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--triton-container",
        type=str,
        required=True,
        help="Triton base container to use for ORT build.",
    )
    parser.add_argument("--ort-version", type=str, required=True, help="ORT version.")
    parser.add_argument(
        "--output", type=str, required=True, help="File to write Dockerfile to."
    )
    parser.add_argument(
        "--enable-gpu", action="store_true", required=False, help="Enable GPU support"
    )
    parser.add_argument(
        "--ort-build-config",
        type=str,
        default="Release",
        choices=["Debug", "Release", "RelWithDebInfo"],
        help="ORT build configuration.",
    )
    parser.add_argument(
        "--target-platform",
        required=False,
        default=None,
        help='Target for build, can be "linux", "windows", "rhel", or "igpu". If not specified, build targets the current platform.',
    )

    parser.add_argument(
        "--cuda-version", type=str, required=False, help="Version for CUDA."
    )
    parser.add_argument(
        "--cuda-home", type=str, required=False, help="Home directory for CUDA."
    )
    parser.add_argument(
        "--cudnn-home", type=str, required=False, help="Home directory for CUDNN."
    )
    parser.add_argument(
        "--ort-openvino",
        type=str,
        required=False,
        help="Enable OpenVino execution provider using specified OpenVINO version.",
    )
    parser.add_argument(
        "--ort-tensorrt",
        action="store_true",
        required=False,
        help="Enable TensorRT execution provider.",
    )
    parser.add_argument(
        "--tensorrt-home", type=str, required=False, help="Home directory for TensorRT."
    )
    parser.add_argument(
        "--onnx-tensorrt-tag", type=str, default="", help="onnx-tensorrt repo tag."
    )
    parser.add_argument("--trt-version", type=str, default="", help="TRT version.")

    # ROCm/MIGraphX arguments
    parser.add_argument(
        "--enable-rocm", action="store_true", required=False, help="Enable ROCm GPU support"
    )
    parser.add_argument(
        "--rocm-version", type=str, required=False, help="Version for ROCm."
    )
    parser.add_argument(
        "--rocm-home", type=str, required=False, help="Home directory for ROCm."
    )
    parser.add_argument(
        "--ort-migraphx",
        action="store_true",
        required=False,
        help="Enable MIGraphX execution provider.",
    )
    parser.add_argument(
        "--onnxruntime-repo",
        type=str,
        default="https://github.com/Microsoft/onnxruntime",
        help="ONNX Runtime (ROCm) git repo URL for build-from-source.",
    )
    parser.add_argument(
        "--onnxruntime-branch",
        type=str,
        default="v1.29.0",
        help="ONNX Runtime (ROCm) git branch for build-from-source.",
    )
    parser.add_argument(
        "--migraphx-repo",
        type=str,
        default="https://github.com/ROCm/AMDMIGraphX.git",
        help="MIGraphX git repo URL for build-from-source.",
    )
    parser.add_argument(
        "--migraphx-branch",
        type=str,
        default="develop",
        help="MIGraphX git branch for build-from-source.",
    )
    parser.add_argument(
        "--migraphx-build-mode",
        type=str,
        choices=["source", "package"],
        default="source",
        help="How to provision MIGraphX for the ROCm build. 'source' builds "
        "--migraphx-branch from source, reusing a prebuilt rocMLIR (no LLVM "
        "recompile). 'package' pulls the prebuilt amdrocm-migraphx packages and "
        "skips the MIGraphX and rocMLIR source builds entirely.",
    )
    parser.add_argument(
        "--migraphx-package-url",
        type=str,
        default="https://stable.repo.amd.com/rocm/migraphx/packages/debian12/pool/main",
        help="Base URL of the prebuilt amdrocm-migraphx .deb pool "
        "(used when --migraphx-build-mode=package).",
    )
    parser.add_argument(
        "--migraphx-package-version",
        type=str,
        default="2.17.0+rocm10.0.0",
        help="Version tag of the prebuilt amdrocm-migraphx packages (used when "
        "--migraphx-build-mode=package). Must match the base image ROCm train "
        "(e.g. 2.17.0 => rocm10.0.0) and Debian release (debian12).",
    )
    parser.add_argument(
        "--rocmlir-repo",
        type=str,
        default="https://github.com/ROCm/rocMLIR.git",
        help="rocMLIR git repo URL. rocMLIR (librockCompiler) is prebuilt once "
        "into /opt/rocmlir and MIGraphX links it instead of rebuilding LLVM/MLIR "
        "from source.",
    )
    parser.add_argument(
        "--rocmlir-commit",
        type=str,
        default="2e1e7abf4ec789e74e49e42018f852ea66e5ef85",
        help="rocMLIR git commit to prebuild. MUST match the ROCm/rocMLIR@<sha> "
        "entry in MIGraphX's requirements.txt for ABI compatibility.",
    )
    parser.add_argument(
        "--migraphx-ep-repo",
        type=str,
        default="https://github.com/onnxruntime/onnxruntime-ep-amdgpu.git",
        help="MIGraphX plugin EP (onnxruntime-ep-amdgpu) git repo URL for "
        "build-from-source.",
    )
    parser.add_argument(
        "--migraphx-ep-branch",
        type=str,
        default="reduce_compute_io_overhead",
        help="MIGraphX plugin EP (onnxruntime-ep-amdgpu) git branch for "
        "build-from-source.",
    )

    FLAGS = parser.parse_args()
    if FLAGS.enable_gpu or FLAGS.enable_rocm:
        preprocess_gpu_flags()

    # if a tag is provided by the user, then simply use it
    # if the tag is empty - check whether there is an entry in the ORT_TO_TRTPARSER_VERSION_MAP
    # map corresponding to ort version + trt version combo. If yes then use it
    # otherwise we leave it empty and use the defaults from ort
    if (
        FLAGS.onnx_tensorrt_tag == ""
        and FLAGS.ort_version in ORT_TO_TRTPARSER_VERSION_MAP.keys()
    ):
        trt_version = re.match(r"^[0-9]+\.[0-9]+", FLAGS.trt_version)
        if (
            trt_version
            and trt_version.group(0)
            == ORT_TO_TRTPARSER_VERSION_MAP[FLAGS.ort_version][0]
        ):
            FLAGS.onnx_tensorrt_tag = ORT_TO_TRTPARSER_VERSION_MAP[FLAGS.ort_version][1]

    if target_platform() == "windows":
        # OpenVINO EP not yet supported for windows build
        if FLAGS.ort_openvino is not None:
            print("warning: OpenVINO not supported for windows, ignoring")
            FLAGS.ort_openvino = None
        dockerfile_for_windows(FLAGS.output)
    else:
        dockerfile_for_linux(FLAGS.output)
