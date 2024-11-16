#pragma once

#include "matrix.hpp"

#include <bitset>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace matrix {

namespace utils {
Fields read_fields(std::string file_path, bool transposed,
                   std::vector<midx_t> *keep_rows,
                   std::vector<midx_t> *keep_cols);

Fields *get_fields(std::shared_ptr<std::vector<char>> serialized_data);

void write_matrix_market(std::string file_path, midx_t height, midx_t width,
                         std::vector<Cell<>> &lines);

template <typename T>
static inline void insertcpy(std::vector<T> &dst, T *src, midx_t amt) {
  dst.insert(dst.end(), src, src + amt);
}

class vector_memory_resource : public std::pmr::memory_resource {
private:
  std::shared_ptr<std::vector<char>> buf;
  size_t offst;

protected:
  void *do_allocate(size_t bytes, size_t alignment) override;

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override;

  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override;

public:
  explicit vector_memory_resource(std::shared_ptr<std::vector<char>> buf);
};

BlockedFields *
get_blocked_fields(std::shared_ptr<std::vector<char>> serialized_data);

} // namespace utils

template <typename T = double> class Cells {
public:
  // Don't use!
  midx_t height;
  midx_t width;
  std::map<CellPos, T> _cells;

private:
  std::vector<midx_t> non_zero_per_row;

public:
  // Takes in the number of rows
  Cells(midx_t height, midx_t width, midx_t non_zeros = 0)
      : height(height), width(width),
        non_zero_per_row(std::vector<midx_t>(height, 0)) {}
  ~Cells() = default;

  // Add one cell to the list of Cells
  // usage: add({row, col}, val)
  // indexes are 0-based
  // If a cell in {row, col} has already been inserted, the value is summed
  // to the previous
  void add(CellPos pos, T val) {
    assert(pos.first < non_zero_per_row.size());
    assert(val != 0);

    if (!_cells.contains(pos))
      ++non_zero_per_row[pos.first];
    else
      val += _cells[pos];

    _cells[pos] = val;
  }

  // Returns the amount of cells (i.e., non_zeros)
  midx_t non_zeros() { return _cells.size(); }

  // Returns the number of cells in a row
  midx_t cells_in_row(midx_t row) {
    assert(row < non_zero_per_row.size());
    return non_zero_per_row[row];
  }
};

template <typename T>
Cells<T> get_cells(std::string file_path, bool transposed,
                   std::vector<midx_t> *keep_rows,
                   std::vector<midx_t> *keep_cols) {
  auto keep_rows_map = std::unordered_map<midx_t, midx_t>();
  if (keep_rows != nullptr)
    for (midx_t i = 0; i < keep_rows->size(); ++i) {
      keep_rows_map.insert({(*keep_rows)[i], i});
    }
  auto keep_cols_map = std::unordered_map<midx_t, midx_t>();
  if (keep_cols != nullptr)
    for (midx_t i = 0; i < keep_cols->size(); ++i) {
      keep_cols_map.insert({(*keep_cols)[i], i});
    }
  bool full = keep_rows == nullptr && keep_cols == nullptr;

  auto fields = utils::read_fields(file_path, transposed, keep_rows, keep_cols);
  std::ifstream stream(file_path);
  std::string sink;
  do {
    getline(stream, sink);
  } while (sink.size() > 0 and sink[0] == '%');

  auto cells = full ? Cells<T>(fields.height, fields.width, fields.non_zeros)
                    : Cells<T>(fields.height, fields.width);

  // read non-zeros
  auto l = fields.non_zeros;
  while (l--) {
    midx_t _row, _col;
    T val;
    stream >> _row;
    stream >> _col;
    stream >> val;

    auto row = transposed ? _col - 1 : _row - 1;
    auto col = transposed ? _row - 1 : _col - 1;

    if (keep_rows != nullptr) {
      if (!keep_rows_map.contains(row))
        continue;

      row = keep_rows_map[row];
    }
    if (keep_cols != nullptr) {
      if (!keep_cols_map.contains(col))
        continue;

      col = keep_cols_map[col];
    }

    cells.add({row, col}, val);
  }
  stream.close();
  return cells;
}

template <typename T = double, class Allocator = std::allocator<char>>
class CSRMatrix {
private:
  std::shared_ptr<std::vector<char, Allocator>> data;
  Fields *fields;

  // Returns the expected size of data in bytes
  size_t expected_data_size() {
    return sizeof(Fields) + ((height + 1) * sizeof(midx_t)) +
           (non_zeros * sizeof(midx_t)) + (non_zeros * sizeof(T));
  }
  std::tuple<midx_t *, midx_t *, T *> get_offsets() {
    assert(data->size() == expected_data_size());
    char *data_ptr = data->data();

    // Make sure things that come after fields are memory-aligned
    assert(sizeof(Fields) % sizeof(midx_t) == 0);

    midx_t *row_ptr = (midx_t *)(data_ptr + sizeof(Fields));
    midx_t *col_idx =
        (midx_t *)(((char *)row_ptr) + ((height + 1) * sizeof(midx_t)));
    T *values = (T *)(((char *)col_idx) + (non_zeros * sizeof(midx_t)));

    assert((midx_t)((char *)row_ptr - data_ptr) == sizeof(Fields));
    assert((midx_t)((char *)col_idx - (char *)row_ptr) ==
           (height + 1) * sizeof(midx_t));
    assert((midx_t)((char *)values - (char *)col_idx) ==
           non_zeros * sizeof(midx_t));

    return {row_ptr, col_idx, values};
  }

public:
  midx_t height;
  midx_t width;
  midx_t non_zeros;
  bool transposed;

  midx_t *row_ptr = nullptr;
  midx_t *col_idx = nullptr;
  T *values = nullptr;

  CSRMatrix(std::string file_path, bool tr = false,
            std::vector<midx_t> *keep_rows = nullptr,
            std::vector<midx_t> *keep_cols = nullptr,
            const Allocator &alloc = Allocator())
      : CSRMatrix(get_cells<T>(file_path, tr, keep_rows, keep_cols), tr) {}

  CSRMatrix(Cells<T> cells, bool tr = false,
            const Allocator &alloc = Allocator())
      : height(cells.height), width(cells.width), non_zeros(cells.non_zeros()),
        transposed(tr) {
#ifndef NDEBUG
    for (auto [pos, _] : cells._cells) {
      auto [row, col] = pos;
      assert(row < height);
      assert(col < width);
    }
#endif

    data = std::make_shared<std::vector<char>>(expected_data_size(), alloc);

    fields = utils::get_fields(data);
    fields->transposed = tr;
    fields->height = height;
    fields->width = width;
    fields->non_zeros = non_zeros;

    auto [_row_ptr, _col_idx, _values] = get_offsets();
    row_ptr = _row_ptr;
    col_idx = _col_idx;
    values = _values;

    row_ptr[0] = 0;
    for (midx_t i = 1; i <= height; ++i) {
      row_ptr[i] = row_ptr[i - 1] + cells.cells_in_row(i - 1);
    }

    // Fill values and col_index arrays using row_ptr
    auto next_pos_in_row = std::vector<midx_t>(height + 1, 0);
    for (auto [pos, val] : cells._cells) {
      auto [row, col] = pos;

      auto index = row_ptr[row] + next_pos_in_row[row];
      col_idx[index] = col;
      values[index] = val;
      next_pos_in_row[row]++;
    }
  }

  CSRMatrix(std::shared_ptr<std::vector<char>> serialized_data)
      : data(serialized_data), fields(utils::get_fields(serialized_data)),
        height(fields->height), width(fields->width),
        non_zeros(fields->non_zeros), transposed(fields->transposed) {
    auto [_row_ptr, _col_idx, _values] = get_offsets();
    row_ptr = _row_ptr;
    col_idx = _col_idx;
    values = _values;
  }

  // Copy constructor
  CSRMatrix(const CSRMatrix &mtx, const Allocator &alloc = Allocator()) {
    data = make_shared<std::vector<char>>(*mtx.data, alloc);
    // Copy over the fields
    height = mtx.height;
    width = mtx.width;
    non_zeros = mtx.non_zeros;
    transposed = mtx.transposed;

    // Adjust the field pointers
    fields = utils::get_fields(data);

    auto [_row_ptr, _col_idx, _values] = get_offsets();
    row_ptr = _row_ptr;
    col_idx = _col_idx;
    values = _values;
  }

  ~CSRMatrix() = default;

  SmallVec<T> row(midx_t i) {
    assert(!transposed);
    assert(i < height);
    auto offst = row_ptr[i];
    return {values + offst, col_idx + offst, row_ptr[i + 1] - offst};
  }
  SmallVec<T> col(midx_t j) {
    assert(transposed);
    assert(j < height);
    auto offst = row_ptr[j];
    return {values + offst, col_idx + offst, row_ptr[j + 1] - offst};
  }

  void save(std::string file_path) {
    auto lines = std::vector<Cell<>>(non_zeros);
    midx_t l = 0;
    for (midx_t row = 0; row < height; ++row) {
      for (midx_t j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
        auto col = col_idx[j];
        if (!transposed)
          lines[l] = {{row, col}, values[j]};
        else
          lines[l] = {{col, row}, values[j]};
        ++l;
      }
    }
    assert(l == non_zeros);

    utils::write_matrix_market(file_path, height, width, lines);
  }

  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<char>> serialize() { return data; }
};

template <typename T = double, class Allocator = std::allocator<char>>
class BlockedCSRMatrix {
private:
  std::shared_ptr<std::vector<char>> data;
  BlockedFields *blocked_fields;

  size_t initial_data_size() { return sizeof(BlockedFields); }

  utils::vector_memory_resource vms;
  Allocator alloc;

  std::vector<CSRMatrix<T, Allocator> *> csrs;
  std::map<midx_t, CSRMatrix<T, Allocator> *> start_row_to_csrs;

  void compute_class_fields() {
    width = csrs[0].width;

#ifndef NDEBUG
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      assert(csrs[i].width == width);
    }
#endif

    for (size_t i; i < blocked_fields->n_sections; ++i) {
      start_row_to_csrs[blocked_fields->section_start_row[i]] = &csrs[i];

      height += csrs[i].height;
      non_zeros += csrs[i].non_zeros;
    }
  }

  void compute_blocked_fields_from_csrs() {
    // Here we assume all sections are filled
    size_t section_height = height / blocked_fields->n_sections;
    char *start = data->data();
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      blocked_fields->section_offst[i] = csrs[i]->data->data() - start;
      blocked_fields->section_start_row[i] = section_height * i;
    }
  }

  void compute_csrs_from_blocked_fields() {
    char *start = data->data();
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      csrs[i] = static_cast<CSRMatrix<T, Allocator> *>(
          static_cast<void *>(&start[blocked_fields->section_offst[i]]));
    }
  }

public:
  midx_t height;
  midx_t width;
  midx_t non_zeros;

  BlockedCSRMatrix(std::string file_path,
                   std::vector<midx_t> *keep_cols = nullptr)
      : blocked_fields(utils::get_blocked_fields(data)), csrs(4, nullptr),
        width(0), height(0), non_zeros(0) {
    data = std::make_shared<std::vector<char>>(initial_data_size());
    blocked_fields = utils::get_blocked_fields(data);

    vms = utils::vector_memory_resource(data);
    alloc = std::pmr::polymorphic_allocator<std::byte>(&vms);

    static_assert(N_SECTIONS > 1);
    blocked_fields->n_sections = N_SECTIONS;

    for (size_t i; i < blocked_fields->n_sections; ++i) {
      csrs[i] = new CSRMatrix<T, Allocator>(file_path, false, nullptr,
                                            keep_cols, alloc);
    }

    compute_class_fields();
    compute_blocked_fields_from_csrs();
  }

  BlockedCSRMatrix(std::shared_ptr<std::vector<char>> serialized_data)
      : data(serialized_data), blocked_fields(utils::get_blocked_fields(data)),
        csrs(4, nullptr), alloc(data), width(0), height(0), non_zeros(0) {
    compute_csrs_from_blocked_fields();
    compute_class_fields();
  }

  ~BlockedCSRMatrix() {
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      free(csrs[i]);
    }
  }

  SmallVec<T> row(midx_t i) {
    auto rel = i % N_SECTIONS;
    auto block_i = i - rel;
    return csrs[block_i]->row(rel);
  }

  CSRMatrix<T, Allocator> *block(midx_t i) {
    assert(i < N_SECTIONS);
    return &csrs[i];
  }

  BlockedCSRMatrix filter(std::bitset<N_SECTIONS> bitmap) {
    auto new_data = std::make_shared<std::vector<char>>(initial_data_size());
    auto bf = utils::get_blocked_fields(data);
    bf->n_sections = bitmap.count();

    auto new_vms = utils::vector_memory_resource(new_data);
    auto new_alloc = std::pmr::polymorphic_allocator<std::byte>(&vms);

    size_t j = 0;
    size_t section_height = height / blocked_fields->n_sections;
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      if (!bitmap[i])
        continue;

      CSRMatrix(csrs[i], new_alloc);
      bf->section_offst[j] = new_data->size();
      bf->section_start_row[j] = blocked_fields->section_start_row[i];
    }

    return BlockedCSRMatrix(new_data);
  }

  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<char>> serialize() { return data; }
};

} // namespace matrix
