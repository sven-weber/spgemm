#pragma once

#include <memory>
#include <string>
#include <vector>

#define ROUND8(x) (((x) + 7) & ~7)

typedef uint32_t midx_t;

namespace matrix {

template <typename T = double> struct SmallVec {
  const T *data;
  const midx_t *pos;
  const midx_t len;
};

using CellPos = std::pair<midx_t, midx_t>;
template <typename T = double>
//                row,    col
using Cell = std::pair<CellPos, T>;

using section = std::pair<midx_t, midx_t>;

typedef struct alignas(8) Fields {
  bool transposed;
  midx_t height;
  midx_t width;
  midx_t non_zeros;
} Fields;

// This is an barebones Matrix class
class Matrix {
private:
  std::shared_ptr<std::vector<std::byte>> raw_data;
  Fields *fields;

  size_t expected_data_size();
  double *get_offset();

  Matrix(Fields fields);

public:
  midx_t height;
  midx_t width;
  bool transposed;

  double *data = nullptr;

  Matrix(midx_t height, midx_t width, bool transposed = false);
  Matrix(std::string file_path, bool transposed = false,
         std::vector<midx_t> *keep_rows = nullptr,
         std::vector<midx_t> *keep_cols = nullptr);
  Matrix(std::shared_ptr<std::vector<std::byte>> serialized_data);
  ~Matrix() = default;

  void save(std::string file_path);
  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<std::byte>> serialize();
};

typedef struct alignas(8) BlockedFields {
  size_t height;
  size_t n_sections;
  size_t section_offst[N_SECTIONS];
  size_t section_start_row[N_SECTIONS];
} BlockedFields;

} // namespace matrix

#include "matrix_impl.hpp"
