#pragma once

#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::CSRMatrix<> &C, int size);
} // namespace baseline
} // namespace parts
