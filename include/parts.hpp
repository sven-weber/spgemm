#pragma once

#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Fields &matrix_fields, int mpi_size);
partition::Partitions partition(midx_t height, midx_t width, int mpi_size);
partition::Partitions balanced_partition(std::string path_A, midx_t width_B,
                                         int mpi_size);
partition::Partitions balanced_partition_square(std::string path_A,
                                                midx_t width_B, int mpi_size);
} // namespace baseline
} // namespace parts
