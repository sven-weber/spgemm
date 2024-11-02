cmake_minimum_required(VERSION 3.9.1)
project(dphpc)

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

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dphpc PUBLIC OpenMP::OpenMP_CXX)
endif()

# MPI
find_package(MPI REQUIRED)
target_link_libraries(dphpc PUBLIC MPI::MPI_CXX)
