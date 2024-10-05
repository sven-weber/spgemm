#pragma once

#include <cstddef>
#include <vector>

namespace partition {

typedef struct Partition {
  size_t start_row;
  size_t end_row;
  size_t start_col;
  size_t end_col;
} Partition;

typedef std::vector<Partition> Partitions;

} // namespace partition