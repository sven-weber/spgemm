#pragma once

#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::CSRMatrix<> &C, int size);
partition::Partitions balanced_partition(matrix::CSRMatrix<> &C, int mpi_size);
} // namespace baseline
} // namespace parts
