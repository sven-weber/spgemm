#pragma once

#include "matrix.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace partition {

typedef struct Partition {
  midx_t start_row;
  midx_t end_row;
  midx_t start_col;
  midx_t end_col;
} Partition;

typedef std::vector<Partition> Partitions;

void save_partitions(Partitions &partitions, std::string file);

typedef std::vector<midx_t> Shuffle;

Shuffle shuffle(size_t size);

Shuffle shuffle_avg(matrix::CSRMatrix<> matrix);

Shuffle shuffle_min(matrix::CSRMatrix<> matrix);

void save_shuffle(Shuffle &shuffle, std::string file);

} // namespace partition
