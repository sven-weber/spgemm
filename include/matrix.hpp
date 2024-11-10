#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace matrix {

typedef struct SmallVec {
  const double *data;
  const size_t *pos;
  const size_t len;
} SmallVec;

//                row,    col
typedef std::pair<size_t, size_t> CellPos;
typedef std::pair<CellPos, double> Cell;

typedef std::pair<int, int> section;

class Cells {
public:
  // Don't use!
  size_t height;
  size_t width;
  std::map<CellPos, double> _cells;

private:
  std::vector<size_t> non_zero_per_row;

public:
  // Takes in the number of rows
  Cells(size_t height, size_t width, size_t non_zeros = 0);
  ~Cells() = default;

  // Add one cell to the list of Cells
  // usage: add({row, col}, val)
  // indexes are 0-based
  // If a cell in {row, col} has already been inserted, the value is summed
  // to the previous
  void add(CellPos pos, double val);

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

  CSRMatrix submatrix(std::vector<section> remove_sections);

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
