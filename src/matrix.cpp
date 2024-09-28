#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "matrix.hpp"

namespace matrix {

struct line {
  size_t row;
  size_t col;
  double val;
};

Matrix::Matrix() {}
/**/
/*double get(size_t, size_t) { return 0; }*/

CSRMatrix::CSRMatrix(std::string file_path, size_t start_i, size_t *,
                     size_t start_j, size_t *)
    : start_i(start_i), start_j(start_j) {
  // TODO: use {start,end}_{i,j}
  std::ifstream stream(file_path);

  if (!stream.is_open()) {
    std::cout << "could not open file: " << file_path << std::endl;
    exit(1);
  }

  std::string line;
  do {
    getline(stream, line);
  } while (line.size() > 0 and line[0] == '%');
  std::istringstream line_stream(line);

  // first non-comment line is going to be:
  // <height> <width> <non_zero>
  line_stream >> height;
  line_stream >> width;
  line_stream >> non_zero;

  values = (double *)malloc(non_zero * sizeof(double));
  col_idx = (size_t *)malloc(non_zero * sizeof(size_t));
  row_ptr = (size_t *)malloc((height + 1) *
                             sizeof(size_t)); // +1 for the extra element

  auto non_zero_per_row = std::vector<size_t>(height, 0);
  auto lines = std::vector<struct line>(non_zero);

  // read non-zeros
  auto l = non_zero;
  while (l--) {
    size_t row, col;
    double val;
    stream >> row;
    stream >> col;
    stream >> val;

    auto i = row - 1;
    auto j = col - 1;
    lines[l] = {i, j, val};

    // number of elements per row
    ++non_zero_per_row[i];
  }

  row_ptr[0] = 0;
  for (size_t i = 1; i <= height; i++) {
    row_ptr[i] = row_ptr[i - 1] + non_zero_per_row[i - 1];
  }

  // Fill values and col_index arrays using row_ptr
  auto next_pos_in_row = std::vector<size_t>(height, 0);
  l = non_zero;
  while (l--) {
    auto [row, col, val] = lines[l];
    auto index = row_ptr[row] + next_pos_in_row[row];
    col_idx[index] = col;
    values[index] = val;
    next_pos_in_row[row]++;
  }

  stream.close();
}

CSRMatrix::~CSRMatrix() {
  if (row_ptr != nullptr)
    free(row_ptr);
  if (col_idx != nullptr)
    free(col_idx);
  if (values != nullptr)
    free(values);
}

double CSRMatrix::get(size_t, size_t) {
  // TODO
  return 0;
}

} // namespace matrix
