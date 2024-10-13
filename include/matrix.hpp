#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
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
  size_t height;
  size_t width;
  size_t non_zeros;

  bool transposed;

  Matrix(size_t height, size_t width, size_t non_zeros,
         bool transposed = false);

  virtual SmallVec row(size_t i) = 0;
  virtual SmallVec col(size_t j) = 0;
};

typedef struct Cell {
  size_t row;
  size_t col;
  double val;

  // sort by column
  bool operator<(const Cell &l) const { return (col < l.col); }

  // pretty printing of cell
  friend std::ostream &operator<<(std::ostream &os, Cell const &c) {
    return os << "(" << c.row << "," << c.row << "," << c.val << ")";
  }
} Cell;

class Cells {
private:
  std::vector<size_t> non_zero_per_row;

public:
  // Don't use!
  std::vector<Cell> _cells;
  size_t height;
  size_t width;

  // Takes in the number of rows
  Cells(size_t height, size_t width, size_t non_zeros = 0);
  ~Cells() = default;

  // Add one cell to the list of Cells
  // usage: add([row, col, val])
  // indexes are 0-based
  void add(Cell &c);

  // Returns the amount of cells (i.e., non_zeros)
  size_t non_zeros();

  // Returns the number of cells in a row
  size_t cells_in_row(size_t row);
};

typedef struct Fields {
  bool transposed;
  size_t height;
  size_t width;
  size_t non_zeros;
} Fields;

class CSRMatrix : public Matrix {
private:
  std::vector<size_t> *keep;
  std::shared_ptr<std::vector<char>> data;
  Fields *fields;

  size_t expected_data_size();
  std::tuple<size_t *, size_t *, double *> get_offsets();

public:
  size_t *row_ptr = nullptr;
  size_t *col_idx = nullptr;
  double *values = nullptr;

  CSRMatrix(std::string file_path, bool transposed = false,
            std::vector<size_t> *keep = nullptr);
  CSRMatrix(Cells cells, bool tranposed = false,
            // This is just for internal use. Should not be used!
            std::vector<size_t> *keep = nullptr);
  CSRMatrix(std::shared_ptr<std::vector<char>> serialized_data);

  SmallVec row(size_t i);
  SmallVec col(size_t j);

  void save(std::string file_path);
  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<char>> serialize();
};

} // namespace matrix
