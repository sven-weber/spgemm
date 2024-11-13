cmake_minimum_required(VERSION 3.9.1)
project(dphpc)
include(FetchContent)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

set(CMAKE_CXX_STANDARD 20)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -Werror -pedantic -lm -O3 -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

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

if(OpenMP_CXX_FOUND)
    target_link_libraries(dphpc PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(dphpc PUBLIC MPI::MPI_CXX)
target_link_libraries(dphpc PUBLIC CombBLAS)
