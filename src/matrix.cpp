#include "matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace matrix {

Cells::Cells(size_t height, size_t width, size_t non_zeros)
    : height(height), width(width),
      non_zero_per_row(std::vector<size_t>(height, 0)) {}

void Cells::add(CellPos pos, double val) {
  assert(pos.first < non_zero_per_row.size());
  assert(val != 0);

  if (!_cells.contains(pos))
    ++non_zero_per_row[pos.first];
  else
    val += _cells[pos];

  _cells[pos] = val;
}

size_t Cells::cells_in_row(size_t row) {
  assert(row < non_zero_per_row.size());
  return non_zero_per_row[row];
}

size_t Cells::non_zeros() { return _cells.size(); }

Fields read_fields(std::string file_path, bool transposed,
                   std::vector<size_t> *keep_rows,
                   std::vector<size_t> *keep_cols) {
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
  if (keep_rows != nullptr)
    height = keep_rows->size();
  if (keep_cols != nullptr)
    width = keep_cols->size();

  // This is going to be overwritten later if we're partially loading the matrix
  line_stream >> non_zeros;

  return {
      .transposed = transposed,
      .height = height,
      .width = width,
      .non_zeros = non_zeros,
  };
  stream.close();
}

Cells get_cells(std::string file_path, bool transposed,
                std::vector<size_t> *keep_rows,
                std::vector<size_t> *keep_cols) {
  auto keep_rows_map = std::unordered_map<size_t, size_t>();
  if (keep_rows != nullptr)
    for (size_t i = 0; i < keep_rows->size(); ++i) {
      keep_rows_map.insert({(*keep_rows)[i], i});
    }
  auto keep_cols_map = std::unordered_map<size_t, size_t>();
  if (keep_cols != nullptr)
    for (size_t i = 0; i < keep_cols->size(); ++i) {
      keep_cols_map.insert({(*keep_cols)[i], i});
    }
  bool full = keep_rows == nullptr && keep_cols == nullptr;

  auto fields = read_fields(file_path, transposed, keep_rows, keep_cols);
  std::ifstream stream(file_path);
  std::string sink;
  do {
    getline(stream, sink);
  } while (sink.size() > 0 and sink[0] == '%');

  Cells cells = full ? Cells(fields.height, fields.width, fields.non_zeros)
                     : Cells(fields.height, fields.width);

  // read non-zeros
  auto l = fields.non_zeros;
  while (l--) {
    size_t _row, _col;
    double val;
    stream >> _row;
    stream >> _col;
    stream >> val;

    auto row = transposed ? _col - 1 : _row - 1;
    auto col = transposed ? _row - 1 : _col - 1;

    if (keep_rows != nullptr) {
      if (!keep_rows_map.contains(row))
        continue;

      row = keep_rows_map[row];
    }
    if (keep_cols != nullptr) {
      if (!keep_cols_map.contains(col))
        continue;

      col = keep_cols_map[col];
    }

    cells.add({row, col}, val);
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

  assert((size_t)((char *)row_ptr - data_ptr) == sizeof(Fields));
  assert((size_t)((char *)col_idx - (char *)row_ptr) ==
         (height + 1) * sizeof(size_t));
  assert((size_t)((char *)values - (char *)col_idx) ==
         non_zeros * sizeof(size_t));

  return {row_ptr, col_idx, values};
}

CSRMatrix::CSRMatrix(std::string file_path, bool transposed,
                     std::vector<size_t> *keep_rows,
                     std::vector<size_t> *keep_cols)
    : CSRMatrix(get_cells(file_path, transposed, keep_rows, keep_cols),
                transposed) {}

CSRMatrix::CSRMatrix(Cells cells, bool transposed)
    : height(cells.height), width(cells.width), non_zeros(cells.non_zeros()),
      transposed(transposed) {
#ifndef NDEBUG
  for (auto [pos, _] : cells._cells) {
    auto [row, col] = pos;
    assert(row < height);
    assert(col < width);
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
  for (auto [pos, val] : cells._cells) {
    auto [row, col] = pos;

    auto index = row_ptr[row] + next_pos_in_row[row];
    col_idx[index] = col;
    values[index] = val;
    next_pos_in_row[row]++;
  }
}

CSRMatrix::CSRMatrix(std::shared_ptr<std::vector<char>> serialized_data)
    : data(serialized_data), fields(get_fields(serialized_data)),
      height(fields->height), width(fields->width),
      non_zeros(fields->non_zeros), transposed(fields->transposed) {
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

void write_matrix_market(std::string file_path, size_t height, size_t width,
                         std::vector<Cell> &lines) {
  std::ofstream stream(file_path);
  if (!stream.is_open()) {
    std::cout << "could not open file (writing): " << file_path << std::endl;
    assert(false);
  }

  stream << "%%MatrixMarket matrix coordinate real general" << std::endl;
  stream << height << " " << width << " " << lines.size() << std::endl;

  std::sort(lines.begin(), lines.end());
  for (auto [pos, val] : lines) {
    auto [row, col] = pos;
    stream << row + 1 << " " << col + 1 << " " << val << std::endl;
  }

  stream.close();
}

void CSRMatrix::save(std::string file_path) {
  auto lines = std::vector<Cell>(non_zeros);
  size_t l = 0;
  for (size_t row = 0; row < height; ++row) {
    for (size_t j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      auto col = col_idx[j];
      if (!transposed)
        lines[l] = {{row, col}, values[j]};
      else
        lines[l] = {{col, row}, values[j]};
      ++l;
    }
  }
  assert(l == non_zeros);

  write_matrix_market(file_path, height, width, lines);
}

template <typename T>
static inline void insertcpy(std::vector<T> &dst, T *src, size_t amt) {
  dst.insert(dst.end(), src, src + amt);
}

CSRMatrix CSRMatrix::submatrix(std::vector<section> remove_sections) {
  auto new_row_ptr = std::vector<size_t>(height + 1);
  auto new_col_idx = std::vector<size_t>();
  auto new_values = std::vector<double>();

  size_t offst = 0;

  auto sec = remove_sections.begin();
  for (size_t i = 0; i < height; ++i) {
    while (i >= sec->second)
      ++sec;

    bool keep = !(i >= sec->first && i < sec->second);
    auto size = row_ptr[i + 1] - row_ptr[i];
    std::cout << "keeping row " << i << " " << keep << " of size " << size
              << std::endl;

    if (keep) {
      new_row_ptr[i] = new_values.size();
      insertcpy(new_col_idx, col_idx + offst + new_values.size(), size);
      insertcpy(new_values, values + offst + new_values.size(), size);
    } else {
      offst += size;
      if (i == 0)
        new_row_ptr[i] = 0;
      else
        new_row_ptr[i] = new_row_ptr[i - 1];
    }
  }
  new_row_ptr[height] = new_values.size();
}

// DO NOT WRITE TO THE OUTPUT OF THIS
std::shared_ptr<std::vector<char>> CSRMatrix::serialize() { return data; }

size_t Matrix::expected_data_size() {
  return sizeof(Fields) +
         ((height * sizeof(size_t)) * (width * sizeof(size_t)));
}

double *Matrix::get_offset() {
  assert(raw_data->size() == expected_data_size());
  char *data_ptr = raw_data->data();

  // Make sure things that come after fields are memory-aligned
  assert(sizeof(Fields) % sizeof(size_t) == 0);
  return (double *)(data_ptr + sizeof(Fields));
}

Matrix::Matrix(size_t height, size_t width, bool transposed)
    : Matrix(Fields{
          .transposed = transposed,
          .height = height,
          .width = width,
          .non_zeros = 0,
      }) {}

// Initializes an empty matrix
Matrix::Matrix(Fields fs)
    : height(fs.height), width(fs.width), transposed(fs.transposed) {
  raw_data = std::make_shared<std::vector<char>>(expected_data_size(), 0);
  fields = get_fields(raw_data);
  memcpy(fields, &fs, sizeof(Fields));
  data = get_offset();
}

#define pos(i, j) ((i) * width + j)

Matrix::Matrix(std::string file_path, bool transposed,
               std::vector<size_t> *keep_rows, std::vector<size_t> *keep_cols)
    : Matrix(read_fields(file_path, transposed, keep_rows, keep_cols)) {
  auto cells = get_cells(file_path, transposed, keep_rows, keep_cols);
  for (auto [pos, val] : cells._cells) {
    auto [row, col] = pos;
    data[pos(row, col)] = val;
  }
}

Matrix::Matrix(std::shared_ptr<std::vector<char>> serialized_data)
    : raw_data(serialized_data), fields(get_fields(serialized_data)),
      height(fields->height), width(fields->width),
      transposed(fields->transposed), data(get_offset()) {}

void Matrix::save(std::string file_path) {
  auto lines = std::vector<Cell>();
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      auto v = data[pos(i, j)];
      if (v != 0) {
        CellPos pos;
        if (!transposed)
          pos = {i, j};
        else
          pos = {j, i};
        lines.push_back({pos, v});
      }
    }
  }

  write_matrix_market(file_path, height, width, lines);
}

// DO NOT WRITE TO THE OUTPUT OF THIS
std::shared_ptr<std::vector<char>> Matrix::serialize() { return raw_data; }

} // namespace matrix
