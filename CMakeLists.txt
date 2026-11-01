cmake_minimum_required(VERSION 3.20)
project(svd VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export symbols on Windows if building shared libraries
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ================= Dependencies =================

# 1. Eigen3 & Boost (Standard CMake imported targets across vcpkg and Nix)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)

# 2. LAPACK & LAPACKE
# Try finding via CMake config first (vcpkg provides lapack/lapacke this way)
find_package(lapack CONFIG QUIET)
find_package(LAPACKE CONFIG QUIET)

if(TARGET lapacke)
    set(LAPACKE_LINK_TARGET lapacke)
else()
    # Fallback to PkgConfig (common on Linux/NixOS)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PKG_LAPACKE QUIET IMPORTED_TARGET lapacke)
    endif()

    if(TARGET PkgConfig::PKG_LAPACKE)
        set(LAPACKE_LINK_TARGET PkgConfig::PKG_LAPACKE)
    else()
        # Fallback to CMake's built-in FindLAPACK + manual header/lib check
        find_package(LAPACK REQUIRED)
        find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h)
        find_library(LAPACKE_LIB NAMES lapacke)

        if(NOT LAPACKE_INCLUDE_DIR OR NOT LAPACKE_LIB)
            message(FATAL_ERROR "LAPACKE headers or libraries not found.")
        endif()

        add_library(lapacke_imported INTERFACE)
        target_include_directories(lapacke_imported INTERFACE ${LAPACKE_INCLUDE_DIR})
        target_link_libraries(lapacke_imported INTERFACE ${LAPACKE_LIB} ${LAPACK_LIBRARIES})
        set(LAPACKE_LINK_TARGET lapacke_imported)
    endif()
endif()

# ================= Target Definitions =================

# Core library
add_library(svd 
    src/givens_refinement.cpp
    src/reverse_jacobi.cpp
    src/naive_mrrr.cpp
    src/newton_polar.cpp
)
target_include_directories(svd PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(svd PUBLIC 
    Eigen3::Eigen 
    Boost::boost 
    ${LAPACKE_LINK_TARGET}
)

# Macro/helper for test executables to reduce repetition
function(add_svd_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE svd)
endfunction()

add_svd_test(svd_test src/testing.cpp)
add_svd_test(svd_test_givref src/givens_refinement.cpp)
add_svd_test(svd_test_revjac src/reverse_jacobi.cpp)
add_svd_test(svd_test_mrrr src/naive_mrrr.cpp)
add_svd_test(svd_test_newton src/newton_polar.cpp)
