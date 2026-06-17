cmake_minimum_required(VERSION 3.19)

# Must be set BEFORE project() so the -mmacosx-version-min flag is emitted.
# torch (conda-forge build) is arm64-only and needs an older libc++ target than
# the host SDK — matches ipt_tilde, without it the libtorch headers fail to compile.
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "" FORCE)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum macOS deployment version" FORCE)

# The Command Line Tools SDK (26.x) ships a libc++ that calls __builtin_clzg, a
# builtin Apple clang 16 lacks — this breaks the libtorch 2.4 headers. Xcode's
# bundled SDK ships a libc++ that matches its clang, so pin the sysroot to it.
if(APPLE AND NOT DEFINED CMAKE_OSX_SYSROOT)
    execute_process(COMMAND xcode-select -p
                    OUTPUT_VARIABLE _xcode_dev OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(_xcode_sdk "${_xcode_dev}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    if(EXISTS "${_xcode_sdk}/usr/include/c++/v1")
        set(CMAKE_OSX_SYSROOT "${_xcode_sdk}" CACHE PATH "" FORCE)
        message(STATUS "Using Xcode SDK sysroot: ${_xcode_sdk}")
    endif()
endif()

project(libipt VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# --------------------------------------------------------------------------------------
# Reuse the ipt_tilde inference core (src/), r8brain, and libtorch — single source
# of truth, nothing is copied. Point IPT_TILDE_DIR at your ipt_tilde checkout.
#
# Local-or-fetch: if a checkout exists (a sibling ../ipt_tilde by default, or wherever
# -DIPT_TILDE_DIR points) it is used as-is — so edits to the core flow straight through,
# which is the co-development workflow. If none is found, a pinned snapshot is fetched
# from GitHub, the same way libtorch is fetched below. Set IPT_TILDE_GIT_TAG to override
# the pin. Only the r8brain submodule is pulled; the Max/pipo submodules aren't needed.
# --------------------------------------------------------------------------------------
include(FetchContent)

set(IPT_TILDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ipt_tilde"
    CACHE PATH "Path to a local ipt_tilde checkout (provides src/, libs/r8brain, libs/libtorch)")
set(IPT_TILDE_GIT_URL "https://github.com/DYCI2/ipt_tilde.git"
    CACHE STRING "ipt_tilde git URL used only when no local checkout is found")
set(IPT_TILDE_GIT_TAG "main"
    CACHE STRING "ipt_tilde git tag/commit fetched when no local checkout is found")

if(EXISTS "${IPT_TILDE_DIR}/src/ipt_classifier.h")
    message(STATUS "ipt_tilde: using local checkout at ${IPT_TILDE_DIR}")
else()
    message(STATUS "ipt_tilde: no local checkout at ${IPT_TILDE_DIR} — fetching ${IPT_TILDE_GIT_TAG}")
    # Download only — do NOT configure ipt_tilde's own root CMakeLists; we consume its
    # sub-targets (libs/r8brain, src/) directly below, exactly as for a local checkout.
    # Pointing SOURCE_SUBDIR at a dir with no CMakeLists.txt makes MakeAvailable populate
    # without calling add_subdirectory (the supported, non-deprecated download-only idiom).
    FetchContent_Declare(ipt_tilde
        GIT_REPOSITORY "${IPT_TILDE_GIT_URL}"
        GIT_TAG        "${IPT_TILDE_GIT_TAG}"
        GIT_SUBMODULES "libs/r8brain/r8brain-free-src"
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  "_libipt_populate_only")
    FetchContent_MakeAvailable(ipt_tilde)
    set(IPT_TILDE_DIR "${ipt_tilde_SOURCE_DIR}" CACHE PATH "" FORCE)
endif()

# --- libtorch (reuse ipt_tilde's local copy, or download to it) ------------------------
set(LIBTORCH_DIR "${IPT_TILDE_DIR}/libs/libtorch")
set(LIBTORCH_PYTHON_VERSION "3.12")
set(LIBTORCH_URL "https://anaconda.org/pytorch/pytorch/2.4.1/download/osx-arm64/pytorch-2.4.1-py3.12_0.tar.bz2")

if(NOT EXISTS "${LIBTORCH_DIR}/lib")
    message(STATUS "libtorch not found at ${LIBTORCH_DIR} — downloading")
    FetchContent_Declare(libtorch
        URL ${LIBTORCH_URL}
        SOURCE_DIR ${LIBTORCH_DIR}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(libtorch)
else()
    message(STATUS "libtorch found locally at ${LIBTORCH_DIR}")
endif()

set(TORCH_CMAKE_DIR "${LIBTORCH_DIR}/lib/python${LIBTORCH_PYTHON_VERSION}/site-packages/torch/share/cmake")
set(TORCH_LIB_DIR   "${LIBTORCH_DIR}/lib/python${LIBTORCH_PYTHON_VERSION}/site-packages/torch/lib")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- shared core targets from ipt_tilde (r8brain static lib + `ipt` INTERFACE lib) -----
add_subdirectory("${IPT_TILDE_DIR}/libs/r8brain" "${CMAKE_BINARY_DIR}/r8brain")
add_subdirectory("${IPT_TILDE_DIR}/src"          "${CMAKE_BINARY_DIR}/ipt_core")

# --------------------------------------------------------------------------------------
# libipt — the C ABI shared library
# --------------------------------------------------------------------------------------
add_library(ipt_api SHARED src/ipt.cpp)
set_target_properties(ipt_api PROPERTIES
    OUTPUT_NAME ipt          # produces libipt.dylib
    # For the in-tree build, find libtorch at its absolute location so build/example
    # runs without an install step. For the installed copy, look right next to
    # libipt.dylib (@loader_path) — the bundled torch dylibs are copied alongside it.
    BUILD_RPATH   "${TORCH_LIB_DIR}"
    INSTALL_RPATH "@loader_path")
target_include_directories(ipt_api PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ipt_api PRIVATE ipt)   # `ipt` brings in src/ headers + r8brain + torch

# --------------------------------------------------------------------------------------
# Example consumer — plain C, demonstrates reuse with zero C++/torch knowledge
# --------------------------------------------------------------------------------------
add_executable(example examples/example.c)
target_link_libraries(example PRIVATE ipt_api)

# --------------------------------------------------------------------------------------
# install / dist — a self-contained, relocatable bundle for external consumers:
#
#     cmake --install build            # -> <source>/dist by default
#     cmake --install build --prefix /somewhere/else
#
# Produces:
#     <prefix>/include/ipt.h
#     <prefix>/lib/libipt.dylib                       (rpath = @loader_path)
#     <prefix>/lib/libtorch.dylib, libtorch_cpu.dylib, libc10.dylib
#
# A consumer compiles with  -I<prefix>/include -L<prefix>/lib -lipt  and, at run
# time, only needs libipt.dylib to find the torch dylibs sitting next to it
# (they already carry @loader_path), so the whole lib/ folder is relocatable.
# --------------------------------------------------------------------------------------
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/dist" CACHE PATH "libipt dist prefix" FORCE)
endif()

# Public header + the C ABI shared library.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/ipt.h" DESTINATION include)
install(TARGETS ipt_api LIBRARY DESTINATION lib)

# The libtorch runtime closure libipt links against (CPU build: no libomp — uses
# Accelerate). These already reference each other via @rpath/@loader_path, so
# co-locating them in lib/ is enough.
install(FILES
    "${TORCH_LIB_DIR}/libtorch.dylib"
    "${TORCH_LIB_DIR}/libtorch_cpu.dylib"
    "${TORCH_LIB_DIR}/libc10.dylib"
    DESTINATION lib)
