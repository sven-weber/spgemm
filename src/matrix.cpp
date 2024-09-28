#include "matrix.hpp"

namespace matrix {

Matrix::Matrix(size_t start_i, size_t height, size_t start_j, size_t width,
               size_t non_zero)
    : start_i(start_i), height(height), start_j(start_j), width(width),
      non_zero(non_zero) {}

CSRMatrix::CSRMatrix(size_t start_i, size_t height, size_t start_j,
                     size_t width, size_t non_zero)
    : Matrix(start_i, height, start_j, width, non_zero) {
  // chris scrivi qua
}

} // namespace matrix
