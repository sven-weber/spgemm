cmake_minimum_required(VERSION 3.9.1)
project(dphpc)
include(FetchContent)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(N_SECTIONS 512 CACHE STRING "Number of sections in which each partition is split")

# OpenMP
find_package(OpenMP)

# MPI
find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "-w")

FetchContent_Declare(
  CombBLAS
  GIT_REPOSITORY https://github.com/lucat1/combblas-spgemm1d
  GIT_TAG        yuxi/spgemm1d
  OVERRIDE_FIND_PACKAGE
)

find_package(CombBLAS)

FetchContent_Declare(
  mio
  GIT_REPOSITORY https://github.com/vimpunk/mio
  OVERRIDE_FIND_PACKAGE
)

find_package(mio)

FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/wjakob/tbb
  OVERRIDE_FIND_PACKAGE
)

set(TBB_BUILD_STATIC CACHE INTERNAL OFF)
set(TBB_BUILD_TBBMALLOC CACHE INTERNAL OFF)
set(TBB_BUILD_TBBMALLOC_PROXY CACHE INTERNAL OFF)
set(TBB_BUILD_TESTS CACHE INTERNAL OFF)
FetchContent_MakeAvailable(TBB)

find_package(TBB)

FetchContent_Declare(
  fast_matrix_market
  GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
  GIT_TAG main
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
)

find_package(fast_matrix_market)

set(CMAKE_CXX_STANDARD 20)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -Werror -pedantic -lm -O3 -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}") #-DNSHUFFLE

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-ggdb -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-DNDEBUG")

set(SOURCE_DIR src)
set(INCLUDE_DIR include)

file(GLOB SOURCES ${SOURCE_DIR}/**/**/*.cpp ${SOURCE_DIR}/**/*.cpp ${SOURCE_DIR}/*.cpp)
file(GLOB HEADERS LIST_DIRECTORIES true ${INCLUDE_DIR})

# message(SOURCES="${SOURCES}")
# message(HEADERS="${HEADERS}")

add_executable(dphpc ${SOURCES})
target_include_directories(dphpc PRIVATE ${HEADERS})

target_include_directories(dphpc PRIVATE
  ${CombBLAS_SOURCE_DIR}/3DSpGEMM
)


if(OpenMP_CXX_FOUND)
    target_link_libraries(dphpc PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(dphpc PUBLIC N_SECTIONS=${N_SECTIONS})

target_link_libraries(dphpc PUBLIC MPI::MPI_CXX)
target_link_libraries(dphpc PUBLIC CombBLAS)
target_link_libraries(dphpc PUBLIC tbb)
target_link_libraries(dphpc PUBLIC mio::mio)
target_link_libraries(dphpc PUBLIC fast_matrix_market::fast_matrix_market)
