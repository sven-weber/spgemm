#include "parts.hpp"

#include <cassert>
#include <iostream>

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::CSRMatrix &C, int mpi_size) {
  partition::Partitions p(mpi_size);

  int rows_per_partition = C.height / mpi_size;
  int n_extra_row = C.height % mpi_size;
  int curr_row = 0;

  int cols_per_partition = C.width / mpi_size;
  int n_extra_col = C.width % mpi_size;
  int curr_col = 0;

  for (int i = 0; i < mpi_size; i++) {
    p[i].start_row = curr_row;
    int tmp_rows = rows_per_partition + (i < n_extra_row);
    p[i].end_row = curr_row + tmp_rows;
    curr_row += tmp_rows;

    p[i].start_col = curr_col;
    int tmp_cols = cols_per_partition + (i < n_extra_col);
    p[i].end_col = curr_col + tmp_cols;
    curr_col += tmp_cols;
  }

  // Simple 1D partitioning
  /*for (int i = 0; i < mpi_size; i++) {
    p[i].start_row = i * C.height / mpi_size;
    p[i].end_row = (i + 1) * C.height / mpi_size;
    p[i].start_col = i * C.width / mpi_size;
    p[i].end_col = (i + 1) * C.width / mpi_size;
  }*/

  return p;
}
} // namespace baseline
} // namespace parts
