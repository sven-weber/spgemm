#pragma once

#include <cstddef>

namespace matrix {

// This is an abstract base class. Use either:
// - CSRMatrix
// - BCSRMatrix
class Matrix {
public:
  size_t height;
  size_t width;

  Matrix(size_t height, size_t width);

  virtual void set(size_t i, size_t j, double val);
  virtual double get(size_t i, size_t j);
};

} // namespace matrix
