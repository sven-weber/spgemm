#pragma once

#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Fields &matrix_fields, int mpi_size);
partition::Partitions partition(midx_t height, midx_t width, int mpi_size);
partition::Partitions balanced_partition(matrix::CSRMatrix<> &C,
                                         int mpi_size);
} // namespace baseline
} // namespace parts
