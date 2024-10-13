#include "parts.hpp"

#include <cassert>
#include <iostream>

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Matrix &C, int mpi_size) {
  partition::Partitions p(mpi_size);

  assert(C.width % mpi_size == 0);
  assert(C.height % mpi_size == 0);

  // Simple 1D partitioning
  for (int i = 0; i < mpi_size; i++) {
    p[i].start_row = i * C.height / mpi_size;
    p[i].end_row = (i + 1) * C.height / mpi_size;
    p[i].start_col = i * C.width / mpi_size;
    p[i].end_col = (i + 1) * C.width / mpi_size;
  }

  return p;
}
} // namespace baseline
} // namespace parts