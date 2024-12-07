#pragma once

#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Fields &matrix_fields, int size);
partition::Partitions balanced_partition(matrix::CSRMatrix<short> &C, int mpi_size);
} // namespace baseline
} // namespace parts
