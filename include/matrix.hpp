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
public:
  // Don't use!
  size_t height;
  size_t width;
  std::vector<Cell> _cells;

private:
  std::vector<size_t> non_zero_per_row;

public:
  // Takes in the number of rows
  Cells(size_t height, size_t width, size_t non_zeros = 0);
  ~Cells() = default;

  // Add one cell to the list of Cells
  // usage: add([row, col, val])
  // indexes are 0-based
  void add(Cell c);

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

class CSRMatrix {
private:
  std::shared_ptr<std::vector<char>> data;
  Fields *fields;

  size_t expected_data_size();
  std::tuple<size_t *, size_t *, double *> get_offsets();

public:
  size_t height;
  size_t width;
  size_t non_zeros;
  bool transposed;

  size_t *row_ptr = nullptr;
  size_t *col_idx = nullptr;
  double *values = nullptr;

  CSRMatrix(std::string file_path, bool transposed = false,
            std::vector<size_t> *keep_rows = nullptr,
            std::vector<size_t> *keep_cols = nullptr);
  CSRMatrix(Cells cells, bool tranposed = false);
  CSRMatrix(std::shared_ptr<std::vector<char>> serialized_data);
  ~CSRMatrix() = default;

  SmallVec row(size_t i);
  SmallVec col(size_t j);

  void save(std::string file_path);
  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<char>> serialize();
};

// This is an barebones Matrix class
class Matrix {
private:
  std::shared_ptr<std::vector<char>> raw_data;
  Fields *fields;

  size_t expected_data_size();
  double *get_offset();

  Matrix(Fields fields);

public:
  size_t height;
  size_t width;
  bool transposed;

  double *data = nullptr;

  Matrix(size_t height, size_t width, bool transposed = false);
  Matrix(std::string file_path, bool transposed = false,
         std::vector<size_t> *keep_rows = nullptr,
         std::vector<size_t> *keep_cols = nullptr);
  Matrix(std::shared_ptr<std::vector<char>> serialized_data);
  ~Matrix() = default;

  void save(std::string file_path);
  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<char>> serialize();
};

} // namespace matrix
