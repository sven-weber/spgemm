#include "matrix.hpp"

#include <algorithm>
#include <cassert>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace matrix {

Cells::Cells(size_t height, size_t width, size_t non_zeros)
    : height(height), width(width),
      non_zero_per_row(std::vector<size_t>(height, 0)) {
  _cells.reserve(non_zeros);
}

void Cells::add(Cell c) {
  assert(c.row < non_zero_per_row.size());

  _cells.push_back(c);
  ++non_zero_per_row[c.row];
}

size_t Cells::cells_in_row(size_t row) {
  assert(row < non_zero_per_row.size());
  return non_zero_per_row[row];
}

size_t Cells::non_zeros() { return _cells.size(); }

Cells get_cells(std::string file_path, bool transposed,
                std::vector<size_t> *keep) {
  auto keep_map = std::unordered_map<size_t, size_t>();
  if (keep != nullptr)
    for (size_t i = 0; i < keep->size(); ++i) {
      keep_map.insert({(*keep)[i], i});
    }

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
  size_t width, height, non_zeros;
  if (transposed) {
    line_stream >> width;
    line_stream >> height;
  } else {
    line_stream >> height;
    line_stream >> width;
  }
  if (keep != nullptr)
    height = keep->size();
  // This is going to be overwritten later if we're partially loading the matrix
  line_stream >> non_zeros;

  Cells cells =
      keep == nullptr ? Cells(height, width, non_zeros) : Cells(height, width);

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

    if (keep == nullptr)
      cells.add(c);
    else if (keep_map.contains(c.row)) {
      c.row = keep_map[c.row];
      cells.add(c);
    }
  }
  stream.close();
  return cells;
}

Fields *get_fields(std::shared_ptr<std::vector<char>> serialized_data) {
  return (Fields *)((void *)serialized_data->data());
}

// Returns the expected size of data in bytes
size_t CSRMatrix::expected_data_size() {
  return sizeof(Fields) + ((height + 1) * sizeof(size_t)) +
         (non_zeros * sizeof(size_t)) + (non_zeros * sizeof(double));
}

std::tuple<size_t *, size_t *, double *> CSRMatrix::get_offsets() {
  assert(data->size() == expected_data_size());
  char *data_ptr = data->data();

  // Make sure things that come after fields are memory-aligned
  assert(sizeof(Fields) % sizeof(size_t) == 0);

  size_t *row_ptr = (size_t *)(data_ptr + sizeof(Fields));
  size_t *col_idx =
      (size_t *)(((char *)row_ptr) + ((height + 1) * sizeof(size_t)));
  double *values = (double *)(((char *)col_idx) + (non_zeros * sizeof(size_t)));

  assert(((char *)row_ptr - data_ptr) == sizeof(Fields));
  assert(((char *)col_idx - (char *)row_ptr) == (height + 1) * sizeof(size_t));
  assert(((char *)values - (char *)col_idx) == non_zeros * sizeof(size_t));

  return {row_ptr, col_idx, values};
}

Matrix::Matrix(size_t height, size_t width, size_t non_zeros, bool transposed)
    : height(height), width(width), non_zeros(non_zeros),
      transposed(transposed) {}

CSRMatrix::CSRMatrix(std::string file_path, bool transposed,
                     std::vector<size_t> *keep)
    : CSRMatrix(get_cells(file_path, transposed, keep), transposed) {}

CSRMatrix::CSRMatrix(Cells cells, bool transposed)
    : Matrix(cells.height, cells.width, cells.non_zeros(), transposed) {
#ifndef NDEBUG
  for (auto c : cells._cells) {
    assert(c.row < height);
    assert(c.col < width);
  }
#endif

  data = std::make_shared<std::vector<char>>(expected_data_size());

  fields = get_fields(data);
  fields->transposed = transposed;
  fields->height = height;
  fields->width = width;
  fields->non_zeros = non_zeros;

  auto [_row_ptr, _col_idx, _values] = get_offsets();
  row_ptr = _row_ptr;
  col_idx = _col_idx;
  values = _values;

  row_ptr[0] = 0;
  for (size_t i = 1; i <= height; ++i) {
    row_ptr[i] = row_ptr[i - 1] + cells.cells_in_row(i - 1);
  }

  // Fill values and col_index arrays using row_ptr
  auto next_pos_in_row = std::vector<size_t>(height + 1, 0);
  for (auto [row, col, val] : cells._cells) {
    auto index = row_ptr[row] + next_pos_in_row[row];
    col_idx[index] = col;
    values[index] = val;
    next_pos_in_row[row]++;
  }
}

CSRMatrix::CSRMatrix(std::shared_ptr<std::vector<char>> serialized_data)
    : Matrix(get_fields(serialized_data)->height,
             get_fields(serialized_data)->width,
             get_fields(serialized_data)->non_zeros,
             get_fields(serialized_data)->transposed),
      fields(get_fields(serialized_data)), data(serialized_data) {
  auto [_row_ptr, _col_idx, _values] = get_offsets();
  row_ptr = _row_ptr;
  col_idx = _col_idx;
  values = _values;
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

  // Why the heck do we have to add this?!
  stream << "%%MatrixMarket matrix coordinate integer general" << std::endl;
  stream << height << " " << width << " " << non_zeros << std::endl;

  auto lines = std::vector<Cell>(non_zeros);
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

// DO NOT WRITE TO THE OUTPUT OF THIS
std::shared_ptr<std::vector<char>> CSRMatrix::serialize() { return data; }

} // namespace matrix
