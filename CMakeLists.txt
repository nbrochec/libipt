cmake_minimum_required(VERSION 3.19)

# Must be set BEFORE project() so the -mmacosx-version-min flag is emitted.
# torch (conda-forge build) is arm64-only and needs an older libc++ target than
# the host SDK — without it the libtorch headers fail to compile.
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

include(FetchContent)

# --------------------------------------------------------------------------------------
# libipt owns the IPT inference core (core/), the r8brain resampler (libs/r8brain),
# and acquires libtorch (libs/libtorch). It depends on NOTHING outside this tree.
# --------------------------------------------------------------------------------------

# --- r8brain (vendored under libs/r8brain/r8brain-free-src) ----------------------------
# Source is committed in-tree. If it is missing (e.g. a partial checkout), fetch a pin.
set(R8BRAIN_VENDORED "${CMAKE_CURRENT_SOURCE_DIR}/libs/r8brain/r8brain-free-src")
if(EXISTS "${R8BRAIN_VENDORED}/CDSPResampler.h")
    message(STATUS "r8brain: using vendored source at ${R8BRAIN_VENDORED}")
    set(R8BRAIN_SRC_DIR "${R8BRAIN_VENDORED}")
else()
    # URL (not GIT) fetch: no per-reconfigure "git fetch" update step, and the
    # source lands inside build/_deps alongside its stamps — so removing libs/
    # without removing build/ can never leave FetchContent in a half-state.
    # Pin: later master reorganized pffft.cpp/r8bbase.cpp out of the root, which
    # breaks the static-lib build. version-6.5 keeps them in the tree root.
    message(STATUS "r8brain: vendored source not found — fetching version-6.5")
    FetchContent_Declare(r8brain_src
        URL "https://github.com/avaneev/r8brain-free-src/archive/refs/tags/version-6.5.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR "_libipt_populate_only")   # populate only; we build it inline below
    FetchContent_MakeAvailable(r8brain_src)
    set(R8BRAIN_SRC_DIR "${r8brain_src_SOURCE_DIR}")
endif()

# --- libtorch (downloaded into libs/libtorch; gitignored) ------------------------------
# Same torch version (2.4.1, CPU) on every platform so TorchScript models never
# need per-platform re-exports. Each platform has its own archive and layout;
# both are normalized behind TORCH_CMAKE_DIR / TORCH_LIB_DIR below.
set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch")
if(WIN32)
    # Official MSVC CPU build. The zip carries a top-level libtorch/ directory.
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.1%2Bcpu.zip")
    set(_libtorch_archive "${CMAKE_BINARY_DIR}/libtorch-download.zip")
    set(TORCH_CMAKE_DIR "${LIBTORCH_DIR}/libtorch/share/cmake")
    set(TORCH_LIB_DIR   "${LIBTORCH_DIR}/libtorch/lib")
else()
    # Anaconda build (osx-arm64): as of 2.4.1, ~20x faster on CPU than the
    # pytorch.org macOS binaries. Extracts straight to bin/ info/ lib/.
    set(LIBTORCH_PYTHON_VERSION "3.12")
    set(LIBTORCH_URL "https://anaconda.org/pytorch/pytorch/2.4.1/download/osx-arm64/pytorch-2.4.1-py3.12_0.tar.bz2")
    set(_libtorch_archive "${CMAKE_BINARY_DIR}/libtorch-download.tar.bz2")
    set(TORCH_CMAKE_DIR "${LIBTORCH_DIR}/lib/python${LIBTORCH_PYTHON_VERSION}/site-packages/torch/share/cmake")
    set(TORCH_LIB_DIR   "${LIBTORCH_DIR}/lib/python${LIBTORCH_PYTHON_VERSION}/site-packages/torch/lib")
endif()

# Plain download+extract (not FetchContent) gated solely on the extracted dir
# existing. FetchContent stores its "populated" stamps under build/ while the
# payload lives in libs/ — so `rm -rf libs` without `rm -rf build` would leave a
# stale stamp and an empty libs/libtorch. Keying purely on EXISTS makes this
# self-healing: if the payload is gone, it is re-fetched regardless of build/.
if(NOT EXISTS "${TORCH_LIB_DIR}")
    message(STATUS "libtorch not found at ${LIBTORCH_DIR} — downloading")
    set(_libtorch_tarball "${_libtorch_archive}")
    file(DOWNLOAD "${LIBTORCH_URL}" "${_libtorch_tarball}"
        SHOW_PROGRESS STATUS _libtorch_dl_status)
    list(GET _libtorch_dl_status 0 _libtorch_dl_code)
    if(NOT _libtorch_dl_code EQUAL 0)
        list(GET _libtorch_dl_status 1 _libtorch_dl_msg)
        file(REMOVE "${_libtorch_tarball}")
        message(FATAL_ERROR "libtorch download failed: ${_libtorch_dl_msg}")
    endif()
    # The conda tarball extracts straight to bin/ info/ lib/ (no top-level dir).
    file(MAKE_DIRECTORY "${LIBTORCH_DIR}")
    file(ARCHIVE_EXTRACT INPUT "${_libtorch_tarball}" DESTINATION "${LIBTORCH_DIR}")
    file(REMOVE "${_libtorch_tarball}")
else()
    message(STATUS "libtorch found locally at ${LIBTORCH_DIR}")
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(MSVC)
    # The torch 2.4 headers need standards conformance and (with their template
    # volume) object sections beyond the COFF default. They also pull in
    # windows.h, whose min/max macros would mangle every std::min/std::max —
    # NOMINMAX suppresses them.
    add_compile_options(/permissive- /EHsc /bigobj)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# --- r8brain static lib ----------------------------------------------------------------
# Built inline (not via add_subdirectory) so the only thing that needs to exist under
# libs/r8brain is the fetched/vendored r8brain-free-src/ — no committed wrapper file.
# example.cpp is the upstream demo and has its own main(); exclude it.
file(GLOB R8BRAIN_SOURCES "${R8BRAIN_SRC_DIR}/*.cpp")
list(REMOVE_ITEM R8BRAIN_SOURCES "${R8BRAIN_SRC_DIR}/example.cpp")
add_library(r8brain STATIC ${R8BRAIN_SOURCES})
# SYSTEM suppresses warnings from the vendored r8brain headers.
target_include_directories(r8brain SYSTEM PUBLIC "${R8BRAIN_SRC_DIR}")

# --------------------------------------------------------------------------------------
# ipt_core — the C++ inference core (header-only INTERFACE lib). Private to libipt;
# only src/ipt.cpp consumes it. Consumers of the library use the C ABI (ipt.h) instead.
# --------------------------------------------------------------------------------------
add_library(ipt_core INTERFACE)
target_include_directories(ipt_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/core")
target_link_libraries(ipt_core INTERFACE r8brain "${TORCH_LIBRARIES}")

# --------------------------------------------------------------------------------------
# libipt — the C ABI shared library (the one public artifact)
# --------------------------------------------------------------------------------------
add_library(ipt_api SHARED src/ipt.cpp)
set_target_properties(ipt_api PROPERTIES
    OUTPUT_NAME ipt)         # produces libipt.dylib / ipt.dll (+ ipt.lib import lib)
# IPT_BUILD switches ipt.h's IPT_API macro to dllexport while building the DLL.
target_compile_definitions(ipt_api PRIVATE IPT_BUILD)
if(APPLE)
    set_target_properties(ipt_api PROPERTIES
        # For the in-tree build, find libtorch at its absolute location so build/example
        # runs without an install step. For the installed copy, look right next to
        # libipt.dylib (@loader_path) — the bundled torch dylibs are copied alongside it.
        BUILD_RPATH   "${TORCH_LIB_DIR}"
        INSTALL_RPATH "@loader_path")
endif()
target_include_directories(ipt_api PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ipt_api PRIVATE ipt_core)

# The torch runtime DLL closure (Windows has no rpath: DLLs are found by sitting
# in the loading executable's directory). The *_jni.dll pair is Java-binding
# dead weight — never loaded by the C++ runtime, so don't ship it.
if(WIN32)
    file(GLOB IPT_TORCH_RUNTIME_DLLS "${TORCH_LIB_DIR}/*.dll")
    list(FILTER IPT_TORCH_RUNTIME_DLLS EXCLUDE REGEX "jni")
endif()

# When built via add_subdirectory (e.g. from ipt_tilde), hand the parent the
# torch runtime locations so it can bundle the closure without duplicating any
# knowledge of the per-platform libtorch layout.
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(IPT_TORCH_LIB_DIR "${TORCH_LIB_DIR}" PARENT_SCOPE)
    if(WIN32)
        set(IPT_TORCH_RUNTIME_DLLS "${IPT_TORCH_RUNTIME_DLLS}" PARENT_SCOPE)
    endif()
endif()

# --------------------------------------------------------------------------------------
# Example consumer — plain C, demonstrates reuse with zero C++/torch knowledge
# --------------------------------------------------------------------------------------
add_executable(example examples/example.c)
target_link_libraries(example PRIVATE ipt_api)
if(MSVC)
    # MSVC's math.h only defines M_PI behind this.
    target_compile_definitions(example PRIVATE _USE_MATH_DEFINES)
endif()

# On Windows the build-tree example finds ipt.dll in its own output directory
# already; the torch DLLs must be copied beside it (no rpath mechanism).
if(WIN32)
    add_custom_command(TARGET example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${IPT_TORCH_RUNTIME_DLLS} "$<TARGET_FILE_DIR:example>")
endif()

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

# Public header + the C ABI shared library. On Windows RUNTIME is ipt.dll and
# ARCHIVE its ipt.lib import library; on macOS LIBRARY is libipt.dylib. All in
# lib/ so the whole folder stays relocatable as one unit.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/ipt.h" DESTINATION include)
install(TARGETS ipt_api
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

if(WIN32)
    # The torch runtime DLL closure (globbed above). DLLs resolve each other by
    # co-location, so lib/ next to ipt.dll is enough.
    install(FILES ${IPT_TORCH_RUNTIME_DLLS} DESTINATION lib)
else()
    # The libtorch runtime closure libipt links against (CPU build: no libomp — uses
    # Accelerate). These already reference each other via @rpath/@loader_path, so
    # co-locating them in lib/ is enough.
    install(FILES
        "${TORCH_LIB_DIR}/libtorch.dylib"
        "${TORCH_LIB_DIR}/libtorch_cpu.dylib"
        "${TORCH_LIB_DIR}/libc10.dylib"
        DESTINATION lib)
endif()
