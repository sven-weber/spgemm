#include "matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace matrix {
namespace utils {
namespace fmm = fast_matrix_market;

Fields read_fields(std::string file_path, bool transposed,
                   std::vector<midx_t> *keep_rows,
                   std::vector<midx_t> *keep_cols) {
  fmm::matrix_market_header h;
  std::ifstream stream(file_path);
  if (!stream.is_open()) {
    std::cout << "could not open file (reading): " << file_path << std::endl;
    assert(false);
  }

  // Numeber of non-zeros is wrong and it's going to be overwritten later if
  // we're partially loading the matrix
  fmm::read_header(stream, h);
  if (transposed) {
    std::swap(h.nrows, h.ncols);
  }
  if (keep_rows != nullptr)
    h.nrows = keep_rows->size();
  if (keep_cols != nullptr)
    h.ncols = keep_cols->size();

  stream.close();
  return {
      .transposed = transposed,
      .height = static_cast<midx_t>(h.nrows),
      .width = static_cast<midx_t>(h.ncols),
      .non_zeros = static_cast<midx_t>(h.vector_length),
  };
}

void write_matrix_market(std::string file_path, midx_t height, midx_t width,
                         std::vector<Cell<>> &lines) {
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

BlockedFields *get_blocked_fields(std::byte *serialized_data) {
  return (BlockedFields *)((void *)serialized_data);
}

} // namespace utils

size_t Matrix::expected_data_size() {
  return ROUND64(sizeof(Fields) +
                ((height * sizeof(midx_t)) * (width * sizeof(midx_t))));
}

double *Matrix::get_offset() {
  assert(raw_data->size() == expected_data_size());
  std::byte *data_ptr = raw_data->data();

  // Make sure things that come after fields are memory-aligned
  assert(sizeof(Fields) % sizeof(midx_t) == 0);
  return (double *)(data_ptr + sizeof(Fields));
}

Matrix::Matrix(midx_t height, midx_t width, bool transposed)
    : Matrix(Fields{
          .transposed = transposed,
          .height = height,
          .width = width,
          .non_zeros = 0,
      }) {}

// Initializes an empty matrix
Matrix::Matrix(Fields fs)
    : height(fs.height), width(fs.width), transposed(fs.transposed) {
  raw_data = std::make_shared<std::vector<std::byte>>(expected_data_size(),
                                                      std::byte(0));
  fields = utils::get_fields(raw_data->data());
  memcpy(fields, &fs, sizeof(Fields));
  data = get_offset();
}

#define pos(i, j) ((i) * width + j)

Matrix::Matrix(std::string file_path, bool transposed,
               std::vector<midx_t> *keep_rows, std::vector<midx_t> *keep_cols)
    : Matrix(utils::read_fields(file_path, transposed, keep_rows, keep_cols)) {
  auto cells = get_cells<double>(file_path, transposed, keep_rows, keep_cols);
  for (int row = 0; row < cells._cells.size(); row++) {
    for (auto [col, val] : cells._cells[row]) {
      data[pos(row, col)] = val.value;
    }
  }
}

Matrix::Matrix(std::shared_ptr<std::vector<std::byte>> serialized_data)
    : raw_data(serialized_data),
      fields(utils::get_fields(serialized_data->data())),
      height(fields->height), width(fields->width),
      transposed(fields->transposed), data(get_offset()) {}

void Matrix::save(std::string file_path) {
  auto lines = std::vector<Cell<>>();
  for (midx_t i = 0; i < height; ++i) {
    for (midx_t j = 0; j < width; ++j) {
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

  utils::write_matrix_market(file_path, height, width, lines);
}

// DO NOT WRITE TO THE OUTPUT OF THIS
std::shared_ptr<std::vector<std::byte>> Matrix::serialize() { return raw_data; }

} // namespace matrix
