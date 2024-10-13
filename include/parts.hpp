#pragma once

#include "matrix.hpp"
#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Matrix &C, int size);
} // namespace baseline
} // namespace parts
