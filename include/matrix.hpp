#pragma once

#include <cstddef>
#include <string>

namespace matrix {

// This is an abstract base class. Use either:
// - CSRMatrix
// - BCSRMatrix
class Matrix {
public:
  Matrix();

  virtual double get(size_t i, size_t j) = 0;
};

class CSRMatrix : Matrix {
public:
  size_t *row_ptr = nullptr;
  size_t *col_idx = nullptr;
  double *values = nullptr;

  size_t start_i;
  size_t height;
  size_t start_j;
  size_t width;
  size_t non_zero;

  CSRMatrix(std::string file_path, size_t start_i = 0, size_t *end_i = nullptr,
            size_t start_j = 0, size_t *end_j = nullptr);
  ~CSRMatrix();

  double get(size_t i, size_t j);
};

} // namespace matrix
