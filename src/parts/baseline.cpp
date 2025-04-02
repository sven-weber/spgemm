#include "parts.hpp"

#include <bitset>
#include <cassert>
#include <cmath>
#include <iostream>

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Fields &matrix_fields, int mpi_size) {
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

  return p;
}

partition::Partitions partition(midx_t height, midx_t width, int mpi_size) {
  partition::Partitions p(mpi_size);

  int rows_per_partition = height / mpi_size;
  int n_extra_row = height % mpi_size;
  int curr_row = 0;

  int cols_per_partition = width / mpi_size;
  int n_extra_col = width % mpi_size;
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

  return p;
}

partition::Partitions balanced_partition(std::string path_A, midx_t width_B,
                                         int mpi_size) {
  partition::Partitions p(mpi_size);

  matrix::Fields fields = matrix::utils::read_fields(path_A, false, nullptr, nullptr);
  midx_t non_zeros_per_partition = fields.non_zeros / mpi_size;

  std::vector<midx_t> non_zeros = matrix::get_row_non_zeros<double>(path_A);


  int non_zero_count = 0;
  int node = 0;
  p[node].start_row = 0;
  measure_point(measure::partition, measure::MeasurementEvent::START);
  for (size_t row = 0; row < fields.height; row++) {
    non_zero_count += non_zeros[row];

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
  std::cout << "Rows: " << fields.height - p[node].start_row
            << " - Non Zero: " << non_zero_count << std::endl;
  p[node].end_row = fields.height;
  measure_point(measure::partition, measure::MeasurementEvent::END);

  int cols_per_partition = width_B / mpi_size;
  int n_extra_col = width_B % mpi_size;
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
