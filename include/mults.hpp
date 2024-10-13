#pragma once

#include "matrix.hpp"
#include "partition.hpp"

namespace mults {
namespace baseline {
matrix::CSRMatrix spgemm(matrix::CSRMatrix &part_A, matrix::CSRMatrix &part_B,
                         int rank, int size, partition::Partitions partitions,
                         std::vector<int> serialized_sizes_B_bytes, int max_size_B_bytes);
}
} // namespace mults