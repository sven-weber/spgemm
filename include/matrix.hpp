#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

typedef uint32_t midx_t;

namespace matrix {

typedef struct SmallVec {
  const double *data;
  const midx_t *pos;
  const midx_t len;
} SmallVec;

//                row,    col
typedef std::pair<midx_t, midx_t> CellPos;
typedef std::pair<CellPos, double> Cell;

typedef std::pair<midx_t, midx_t> section;

class Cells {
public:
  // Don't use!
  midx_t height;
  midx_t width;
  std::map<CellPos, double> _cells;

private:
  std::vector<midx_t> non_zero_per_row;

public:
  // Takes in the number of rows
  Cells(midx_t height, midx_t width, midx_t non_zeros = 0);
  ~Cells() = default;

  // Add one cell to the list of Cells
  // usage: add({row, col}, val)
  // indexes are 0-based
  // If a cell in {row, col} has already been inserted, the value is summed
  // to the previous
  void add(CellPos pos, double val);

  // Returns the amount of cells (i.e., non_zeros)
  midx_t non_zeros();

  // Returns the number of cells in a row
  midx_t cells_in_row(midx_t row);
};

typedef struct Fields {
  bool transposed;
  midx_t height;
  midx_t width;
  midx_t non_zeros;
} Fields;

class CSRMatrix {
private:
  std::shared_ptr<std::vector<char>> data;
  Fields *fields;

  midx_t expected_data_size();
  std::tuple<midx_t *, midx_t *, double *> get_offsets();

public:
  midx_t height;
  midx_t width;
  midx_t non_zeros;
  bool transposed;

  midx_t *row_ptr = nullptr;
  midx_t *col_idx = nullptr;
  double *values = nullptr;

  CSRMatrix(std::string file_path, bool transposed = false,
            std::vector<midx_t> *keep_rows = nullptr,
            std::vector<midx_t> *keep_cols = nullptr);
  CSRMatrix(Cells cells, bool tranposed = false);
  CSRMatrix(std::shared_ptr<std::vector<char>> serialized_data);
  ~CSRMatrix() = default;

  SmallVec row(midx_t i);
  SmallVec col(midx_t j);

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

  midx_t expected_data_size();
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
  Matrix(std::shared_ptr<std::vector<char>> serialized_data);
  ~Matrix() = default;

  void save(std::string file_path);
  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<char>> serialize();
};

class BlockedCSRMatrix {
public:
  const midx_t height;
  const midx_t width;
  const midx_t non_zeros;

  BlockedCSRMatrix(std::string file_path, midx_t sections,
                   std::vector<midx_t> *keep_cols = nullptr);
  BlockedCSRMatrix(std::shared_ptr<std::vector<char>> serialized_data);
  ~BlockedCSRMatrix() = default;

  SmallVec row(midx_t i);
  CSRMatrix block(midx_t i);

  BlockedCSRMatrix filter(std::vector<midx_t> keep_blocks);

  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<char>> serialize();
};

} // namespace matrix
