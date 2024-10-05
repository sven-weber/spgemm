#include "matrix.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace matrix {

Cells::Cells(size_t rows, size_t non_zeros)
    : min_i(std::numeric_limits<int64_t>::max()),
      max_i(std::numeric_limits<int64_t>::min()),
      min_j(std::numeric_limits<int64_t>::max()),
      max_j(std::numeric_limits<int64_t>::min()), rows(rows),
      non_zero_per_row(std::vector<size_t>(rows, 0)),
      _cells(std::vector<Cell>(non_zeros)) {}

void Cells::add(Cell &c) {
  assert(c.row < non_zero_per_row.size());

  _cells.push_back(c);
  ++non_zero_per_row[c.row];

  min_i = std::min(min_i, (int64_t)c.row);
  max_i = std::max(max_i, (int64_t)c.row);
  min_j = std::min(min_j, (int64_t)c.col);
  max_j = std::max(max_j, (int64_t)c.col);
}

size_t Cells::cells_in_row(size_t row) {
  assert(row < non_zero_per_row.size());
  return non_zero_per_row[row];
}

size_t Cells::size() { return _cells.size(); }

size_t Cells::height() { return (size_t)(max_i - min_i + 1); }

size_t Cells::width() { return (size_t)(max_j - min_j + 1); }

// prt = malloc(actual_size_i_nedd + sizeof(fields))
//
// actual_data =  ptr + sizeof(fields);
// fields_ptr =  (fields *) ptr + sizeof(fields);
// fields_ptr->height =

Cells get_cells(std::string file_path, bool transposed,
                std::unordered_set<size_t> *keep) {
  std::ifstream stream(file_path);
  if (!stream.is_open()) {
    std::cout << "could not open file (reading): " << file_path << std::endl;
    assert(false);
  }

  std::string line;
  do {
    getline(stream, line);
  } while (line.size() > 0 and line[0] == '%');
  std::istringstream line_stream(line);

  // first non-comment line is going to be:
  // <height> <width> <non_zero>
  size_t useless, height, non_zeros;
  if (transposed) {
    line_stream >> useless;
    line_stream >> height;
  } else {
    line_stream >> useless;
    line_stream >> height;
  }
  // This is going to be overwritten later if we're partially loading the matrix
  line_stream >> non_zeros;

  Cells cells = keep == nullptr ? Cells(height) : Cells(height, non_zeros);

  // read non-zeros
  auto l = non_zeros;
  while (l--) {
    size_t row, col;
    double val;
    stream >> row;
    stream >> col;
    stream >> val;

    Cell c = {.row = transposed ? col - 1 : row - 1,
              .col = transposed ? row - 1 : col - 1,
              .val = val};
    if (keep == nullptr) {
      cells.add(c);
    } else if (keep->contains(transposed ? col - 1 : row - 1)) {
      cells.add(c);
    }
  }
  stream.close();
  return cells;
}

Matrix::Matrix(bool transposed)
    : transposed(transposed), height(0), width(0), non_zeros(0) {}

CSRMatrix::CSRMatrix(std::string file_path, bool transposed,
                     std::unordered_set<size_t> *keep)
    : CSRMatrix(get_cells(file_path, transposed, keep), transposed) {}

CSRMatrix::CSRMatrix(Cells cells, bool transposed) : Matrix(transposed) {
  // These values already take tranposed into account
  height = cells.height();
  width = cells.width();
  non_zeros = cells.size();

  auto csr_size = (non_zeros * sizeof(double)) + (non_zeros * sizeof(size_t)) +
                  (height + 1) * sizeof(size_t);
  _ptr = malloc(sizeof(Fields) + csr_size);

  col_idx = (size_t *)(((char *)_ptr) + sizeof(Fields));
  row_ptr = (size_t *)(((char *)col_idx) + ((height + 1) * sizeof(size_t)));
  values = (double *)(((char *)row_ptr) + (non_zeros * sizeof(size_t)));

  row_ptr[0] = 0;
  for (size_t i = 1; i <= height; ++i) {
    row_ptr[i] = row_ptr[i - 1] + cells.cells_in_row(i - 1);
  }

  // Fill values and col_index arrays using row_ptr
  auto next_pos_in_row = std::vector<size_t>(height, 0);
  for (auto [row, col, val] : cells._cells) {
    auto index = row_ptr[row] + next_pos_in_row[row];
    col_idx[index] = col;
    values[index] = val;
    next_pos_in_row[row]++;
  }
}

CSRMatrix::~CSRMatrix() {
  if (_ptr != nullptr)
    free(_ptr);
  _ptr = nullptr;
  row_ptr = nullptr;
  col_idx = nullptr;
  values = nullptr;
}

SmallVec CSRMatrix::row(size_t i) {
  assert(!transposed);
  assert(i < height);
  auto offst = row_ptr[i];
  return {values + offst, col_idx + offst, row_ptr[i + 1] - offst};
}

SmallVec CSRMatrix::col(size_t j) {
  assert(transposed);
  assert(j < height);
  auto offst = row_ptr[j];
  return {values + offst, col_idx + offst, row_ptr[j + 1] - offst};
}

void CSRMatrix::save(std::string file_path) {
  std::ofstream stream(file_path);
  if (!stream.is_open()) {
    std::cout << "could not open file (writing): " << file_path << std::endl;
    assert(false);
  }

  stream << height << " " << width << " " << non_zeros << std::endl;

  auto lines = std::vector<struct Cell>(non_zeros);
  size_t l = 0;
  for (size_t row = 0; row < height; ++row) {
    for (size_t j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      auto col = col_idx[j];
      if (!transposed)
        lines[l] = {row, col, values[j]};
      else
        lines[l] = {col, row, values[j]};
      ++l;
    }
  }
  assert(l == non_zeros);

  std::sort(lines.begin(), lines.end());
  for (auto line : lines) {
    stream << line.row + 1 << " " << line.col + 1 << " " << line.val
           << std::endl;
  }

  stream.close();
}

} // namespace matrix
