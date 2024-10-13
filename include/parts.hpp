#pragma once

#include "matrix.hpp"
#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Matrix &C, int size);
}
namespace shuffle {
void set_seed(bool reproducible);
std::vector<size_t> shuffle(int size);
} // namespace shuffle
} // namespace parts