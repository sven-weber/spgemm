#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace partition {

typedef struct Partition {
  size_t start_row;
  size_t end_row;
  size_t start_col;
  size_t end_col;
} Partition;

typedef std::vector<Partition> Partitions;

void save_partitions(Partitions &partitions, std::string file);

typedef std::vector<size_t> Shuffle;

Shuffle shuffle(size_t size);

void save_shuffle(Shuffle &shuffle, std::string file);

} // namespace partition
