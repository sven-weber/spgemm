#include "matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory_resource>
#include <sstream>
#include <vector>

namespace matrix {
namespace utils {

Fields read_fields(std::string file_path, bool transposed,
                   std::vector<midx_t> *keep_rows,
                   std::vector<midx_t> *keep_cols) {
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
  midx_t width, height, non_zeros;
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

void *vector_memory_resource::do_allocate(size_t bytes, size_t alignment) {
  assert(offst == buf->size());
  size_t aligned_offst = (offst + alignment - 1) & ~(alignment - 1);

  if (aligned_offst + bytes > buf->size()) {
    size_t new_cap = std::max(buf->size() * 2, aligned_offst + bytes);
    buf->resize(new_cap);
  }

  void *result = buf->data() + aligned_offst;
  offst = aligned_offst + bytes;
  assert(offst == buf->size());
  return result;
}

void vector_memory_resource::do_deallocate(void *ptr, size_t bytes,
                                           size_t alignment) {
  assert(offst == buf->size());
  size_t dealloc_offst = static_cast<char *>(ptr) - buf->data();
  if (dealloc_offst + bytes == offst) {
    offst = dealloc_offst;
    buf->resize(offst);
    // buf->shrink_to_fit();
  }
  assert(offst == buf->size());
}

// Override for equality comparison
bool vector_memory_resource::do_is_equal(
    const std::pmr::memory_resource &other) const noexcept {
  return this == &other;
}

vector_memory_resource::vector_memory_resource(
    std::shared_ptr<std::vector<char>> buf)
    : buf(buf), offst(buf->size()) {}

BlockedFields *
get_blocked_fields(std::shared_ptr<std::vector<char>> serialized_data) {
  return (BlockedFields *)((void *)serialized_data->data());
}

} // namespace utils

size_t Matrix::expected_data_size() {
  return sizeof(Fields) +
         ((height * sizeof(midx_t)) * (width * sizeof(midx_t)));
}

double *Matrix::get_offset() {
  assert(raw_data->size() == expected_data_size());
  char *data_ptr = raw_data->data();

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
  raw_data = std::make_shared<std::vector<char>>(expected_data_size(), 0);
  fields = utils::get_fields(raw_data);
  memcpy(fields, &fs, sizeof(Fields));
  data = get_offset();
}

#define pos(i, j) ((i) * width + j)

Matrix::Matrix(std::string file_path, bool transposed,
               std::vector<midx_t> *keep_rows, std::vector<midx_t> *keep_cols)
    : Matrix(utils::read_fields(file_path, transposed, keep_rows, keep_cols)) {
  auto cells = get_cells<double>(file_path, transposed, keep_rows, keep_cols);
  for (auto [pos, val] : cells._cells) {
    auto [row, col] = pos;
    data[pos(row, col)] = val;
  }
}

Matrix::Matrix(std::shared_ptr<std::vector<char>> serialized_data)
    : raw_data(serialized_data), fields(utils::get_fields(serialized_data)),
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
std::shared_ptr<std::vector<char>> Matrix::serialize() { return raw_data; }

} // namespace matrix
