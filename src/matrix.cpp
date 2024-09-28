#include <algorithm>
#include <cassert>
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

  // sort by column
  bool operator<(const line &l) const { return (col < l.col); }
};

Matrix::Matrix() {}

CSRMatrix::CSRMatrix(std::string file_path, size_t start_i, size_t *,
                     size_t start_j, size_t *)
    : start_i(start_i), start_j(start_j) {
  // TODO: use {start,end}_{i,j}
  std::ifstream stream(file_path);
  if (!stream.is_open()) {
    std::cout << "could not open file (reading): " << file_path << std::endl;
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
  for (size_t i = 1; i <= height; ++i) {
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
  row_ptr = nullptr;
  if (col_idx != nullptr)
    free(col_idx);
  col_idx = nullptr;
  if (values != nullptr)
    free(values);
  values = nullptr;
}

double CSRMatrix::get(size_t, size_t) {
  // TODO
  return 0;
}

void CSRMatrix::save(std::string file_path) {
  std::ofstream stream(file_path);
  if (!stream.is_open()) {
    std::cout << "could not open file (writing): " << file_path << std::endl;
    exit(2);
  }

  stream << height << " " << width << " " << non_zero << std::endl;

  auto lines = std::vector<struct line>(non_zero);
  size_t l = 0;
  for (size_t row = 0; row < height; ++row) {
    for (size_t j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      auto col = col_idx[j];
      lines[l] = {row, col, values[j]};
      ++l;
    }
  }
  assert(l == non_zero);

  std::sort(lines.begin(), lines.end());
  for (auto line : lines) {
    stream << line.row + 1 << " " << line.col + 1 << " " << line.val
           << std::endl;
  }

  stream.close();
}

} // namespace matrix
