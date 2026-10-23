cmake_minimum_required(VERSION 3.12)

project(svd VERSION 1.0.0 LANGUAGES C CXX DESCRIPTION "Implementations of experimental SVD algorithms")

# ========== Build Configuration ===========

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")

set(THREADS 4 CACHE STRING "Number of threads for running tests")

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# ======= Build Artifacts Directory ========

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(DEFAULT_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build")
else()
    set(DEFAULT_OUTPUT_DIR "${CMAKE_BINARY_DIR}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${DEFAULT_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${DEFAULT_OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${DEFAULT_OUTPUT_DIR}")

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${DEFAULT_OUTPUT_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${DEFAULT_OUTPUT_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${DEFAULT_OUTPUT_DIR}")
endforeach()

# Generate config.h in build directory
configure_file(src/config.h.in config.h)

# ============== Dependencies ==============

find_package(Threads REQUIRED)

find_package(Eigen3 3.3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 REQUIRED)
endif()

find_package(Boost REQUIRED)

find_package(lapacke QUIET CONFIG)
if(TARGET lapacke)
    set(LAPACKE_TARGET lapacke)
else()
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PKG_LAPACKE QUIET IMPORTED_TARGET lapacke)
    endif()
    if(TARGET PkgConfig::PKG_LAPACKE)
        set(LAPACKE_TARGET PkgConfig::PKG_LAPACKE)
    else()
        find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h)
        find_library(LAPACKE_LIB NAMES lapacke)
        find_package(LAPACK REQUIRED)
        find_package(BLAS REQUIRED)
        if(LAPACKE_INCLUDE_DIR AND LAPACKE_LIB)
            add_library(lapacke_custom INTERFACE)
            target_include_directories(lapacke_custom INTERFACE ${LAPACKE_INCLUDE_DIR})
            target_link_libraries(lapacke_custom INTERFACE ${LAPACKE_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
            set(LAPACKE_TARGET lapacke_custom)
        else()
            message(FATAL_ERROR "Could not find LAPACKE header (lapacke.h) or library.")
        endif()
    endif()
endif()

# ============ Framework Files =============

set(HEADER_FILES
    # Tests
    src/lib/tests/generate_svd.h
    src/lib/tests/SVD_Test.h
    src/lib/tests/SVD_Test.hpp
    src/lib/tests/SVD_Test_config.h

    # Parfenov's & Drozdov's ideas
    src/lib/idea1_givref/givens_refinement.h
    src/lib/idea1_givref/givens_refinement.hpp
    src/lib/idea2_revjac/reverse_jacobi.h
    src/lib/idea2_revjac/reverse_jacobi.hpp
    src/lib/idea2_revjac/pseudo_reverse_jacobi.h
    src/lib/idea2_revjac/pseudo_reverse_jacobi.hpp
    src/lib/idea3_naive/naive_mrrr/naive_mrrr.h
    src/lib/idea3_naive/naive_mrrr/naive_mrrr.hpp
    src/lib/idea3_naive/naive_bidiag_svd/naive_bidiag_svd.h
    src/lib/idea3_naive/naive_bidiag_svd/naive_bidiag_svd.hpp

    # Legacy
    src/lib/idea1_givref/legacy/v0_givens_refinement.h
    src/lib/idea1_givref/legacy/v0_givens_refinement.hpp
    src/lib/idea1_givref/legacy/v1_givens_refinement.h
    src/lib/idea1_givref/legacy/v1_givens_refinement.hpp
    src/lib/idea2_revjac/legacy/v0_reverse_jacobi.h
    src/lib/idea2_revjac/legacy/v0_reverse_jacobi.hpp
    src/lib/idea3_naive/naive_mrrr/legacy/v0_naive_mrrr.h
    src/lib/idea3_naive/naive_mrrr/legacy/v0_naive_mrrr.hpp

    # Umbrella header
    src/lib/SVD_project.h
)

# Stub translation unit in binary dir so the static library compiles
set(SVD_LIB_STUB "${CMAKE_CURRENT_BINARY_DIR}/svd_stub.cpp")
if(NOT EXISTS "${SVD_LIB_STUB}")
    file(WRITE "${SVD_LIB_STUB}" "// SVD static library translation unit\n")
endif()

# ============= Build Targets ==============

# Static library target: svd
add_library(svd STATIC ${SVD_LIB_STUB} ${HEADER_FILES})
target_include_directories(svd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(svd PUBLIC
    Eigen3::Eigen
    Boost::boost
    ${LAPACKE_TARGET}
    Threads::Threads
)
target_compile_features(svd PUBLIC cxx_std_20)

# Wrapper function to configure test binaries
function(svd_add_test_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE svd)
    set_target_properties(${TARGET_NAME} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

svd_add_test_executable(svd_test src/testing.cpp)
svd_add_test_executable(svd_test_givref src/givens_refinement.cpp)
svd_add_test_executable(svd_test_revjac src/reverse_jacobi.cpp)
