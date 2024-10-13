#pragma once

#include "matrix.hpp"
#include "partition.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Matrix &C, int size);
}
namespace shuffle {
int *shuffle(int size);
}
} // namespace parts