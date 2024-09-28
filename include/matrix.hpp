#pragma once

#include <cstddef>

namespace matrix {

// This is an abstract base class. Use either:
// - CSRMatrix
// - BCSRMatrix
class Matrix {
public:
  size_t start_i;
  size_t height;
  size_t start_j;
  size_t width;
  size_t non_zero;

  Matrix(size_t start_i, size_t height, size_t start_j, size_t width,
         size_t non_zero);

  virtual void set(size_t i, size_t j, double val);
  virtual double get(size_t i, size_t j);
};

class CSRMatrix : Matrix {
private:
  size_t *row_start;
  size_t *col_idx;
  double *values;

public:
  CSRMatrix(size_t start_i, size_t height, size_t start_j, size_t width,
            size_t non_zero);
  ~CSRMatrix();

  void set(size_t i, size_t j, double val);
  double get(size_t i, size_t j);
};

} // namespace matrix
