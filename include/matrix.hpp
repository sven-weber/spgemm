#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace matrix {

typedef struct SmallVec {
  const double *data;
  const size_t *pos;
  const size_t len;
} SmallVec;

// This is an abstract base class. Use either:
// - CSRMatrix
// - BCSRMatrix
class Matrix {
public:
  bool transposed;
  size_t start_i;
  size_t height;
  size_t start_j;
  size_t width;
  size_t non_zero;

  Matrix(std::string file_path, bool transposed = false, size_t start_i = 0,
         size_t *end_i = nullptr, size_t start_j = 0, size_t *end_j = nullptr);

  virtual SmallVec row(size_t i) = 0;
  virtual SmallVec col(size_t j) = 0;
};

class CSRMatrix : public Matrix {
private:
  std::vector<double> _row_ptr;
  std::vector<double> _col_idx;
  std::vector<double> _values;

public:
  size_t *row_ptr = nullptr;
  size_t *col_idx = nullptr;
  double *values = nullptr;

  CSRMatrix(std::string file_path, bool transposed = false, size_t start_i = 0,
            size_t *end_i = nullptr, size_t start_j = 0,
            size_t *end_j = nullptr);
  ~CSRMatrix();

  SmallVec row(size_t i);
  SmallVec col(size_t j);

  void save(std::string file_path);
};

} // namespace matrix
