#pragma once

#include "matrix.hpp"
#include "partition.hpp"

namespace mults {
namespace baseline {
matrix::CSRMatrix spgemm(matrix::Matrix &part_A, matrix::Matrix &part_B,
                         int rank, int size, partition::Partitions partitions);
}
} // namespace mults