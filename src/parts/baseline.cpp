#include "parts.hpp"

#include <bitset>
#include <cassert>
#include <cmath>
#include <iostream>

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Fields matrix_fields, int mpi_size) {
  partition::Partitions p(mpi_size);

  int rows_per_partition = matrix_fields.height / mpi_size;
  int n_extra_row = matrix_fields.height % mpi_size;
  int curr_row = 0;

  int cols_per_partition = matrix_fields.width / mpi_size;
  int n_extra_col = matrix_fields.width % mpi_size;
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

partition::Partitions balanced_partition(matrix::CSRMatrix<short> &C, int mpi_size) {
  partition::Partitions p(mpi_size);

  midx_t non_zeros_per_partition = C.non_zeros / mpi_size;

  int section_width = ceil((float)C.width / (float)N_SECTIONS);
  int non_zero_count = 0;
  int node = 0;
  p[node].start_row = 0;
  for (size_t row = 0; row < C.height; row++) {
    auto [row_data, row_pos, row_len] = C.row(row);
    non_zero_count += row_len;

    if (non_zero_count > non_zeros_per_partition) {
      std::cout << "Rows: " << row - p[node].start_row
                << " - Non Zero: " << non_zero_count << std::endl;
      non_zero_count = 0;
      p[node].end_row = row;
      p[node + 1].start_row = row;
      node++;

      if (node == mpi_size)
        break;
    }
  }
  std::cout << "Rows: " << C.height - p[node].start_row
            << " - Non Zero: " << non_zero_count << std::endl;
  p[node].end_row = C.height;

  int cols_per_partition = C.width / mpi_size;
  int n_extra_col = C.width % mpi_size;
  int curr_col = 0;

  for (int i = 0; i < mpi_size; i++) {
    p[i].start_col = curr_col;
    int tmp_cols = cols_per_partition + (i < n_extra_col);
    p[i].end_col = curr_col + tmp_cols;
    curr_col += tmp_cols;
  }

  return p;
}
} // namespace baseline
} // namespace parts
