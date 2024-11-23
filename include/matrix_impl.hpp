#pragma once

#include "matrix.hpp"

#include <bitset>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

namespace matrix {

namespace utils {
Fields read_fields(std::string file_path, bool transposed,
                   std::vector<midx_t> *keep_rows,
                   std::vector<midx_t> *keep_cols);

template <class T = std::vector<std::byte>>
Fields *get_fields(std::shared_ptr<T> serialized_data) {
  return (Fields *)((void *)serialized_data->data());
}

void write_matrix_market(std::string file_path, midx_t height, midx_t width,
                         std::vector<Cell<>> &lines);

template <typename T>
static inline void insertcpy(std::vector<T> &dst, T *src, midx_t amt) {
  dst.insert(dst.end(), src, src + amt);
}

BlockedFields *
get_blocked_fields(std::shared_ptr<std::vector<std::byte>> serialized_data);

template <typename T> class ContiguousAllocator {
private:
  std::shared_ptr<std::vector<T, std::allocator<T>>> data;
  T *begin() { return data->data(); }
  T *end() { return data->data() + data->size(); }

public:
  using value_type = T;

  ContiguousAllocator() = delete;
  /*{*/
  /*  assert(false && "ContiguousAllocator cannot be default-instantiated");*/
  /*}*/
  ContiguousAllocator(std::shared_ptr<std::vector<std::byte>> &data)
      : data(data) {}

  template <typename U>
  constexpr ContiguousAllocator(const ContiguousAllocator<U> &c) noexcept {
    std::cout << "CoPy COnSTruCtoR" << std::endl;
    c.data = c.data;
  }

  T *allocate(size_t n) {
    std::cout << "allocating " << n << " types" << std::endl;
    std::cout << "before allocation, vector size: " << data->size()
              << ", vector pointer: " << data->data() << std::endl;
    if (n > std::allocator_traits<ContiguousAllocator>::max_size(*this)) {
      throw std::bad_alloc();
    }
    T *ptr = end();
    data->resize(data->size() + n);
    std::cout << "after allocation, vector size: " << data->size()
              << " vector pointer: " << data->data() << std::endl;
    return ptr;
  }

  void deallocate(T *p, std::size_t n) noexcept {
    std::cout << "deallocating " << n << " types" << std::endl;
    if (p < begin() || p + n > end()) {
      throw std::bad_alloc();
    }
    // We can only de-allocate from the end;
    assert(p + n == end());
    data->resize(data->size() - n);
  }

  /*template <typename U, typename... Args> void construct(U *p, Args &&...args)
   * {*/
  /*  new (p) U(std::forward<Args>(args)...);*/
  /*}*/
  /**/
  /*template <typename U> void destroy(U *p) noexcept { p->~U(); }*/

  friend bool operator==(const ContiguousAllocator &a,
                         const ContiguousAllocator &b) {
    return a.data == b.data;
  }
  friend bool operator!=(const ContiguousAllocator &a,
                         const ContiguousAllocator &b) {
    return a.data != b.data;
  }
};

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

template <typename T = double, class Allocator = std::allocator<std::byte>>
class CSRMatrix {
private:
  std::shared_ptr<std::vector<std::byte, Allocator>> data;
  Fields *fields;

  size_t expected_data_size() {
    return sizeof(Fields) + ((height + 1) * sizeof(midx_t)) +
           (non_zeros * sizeof(midx_t)) + (non_zeros * sizeof(T));
  }

  // Returns the expected size of data in bytes
  std::tuple<midx_t *, midx_t *, T *> get_offsets() {
    assert(data->size() == expected_data_size());
    std::byte *data_ptr = data->data();

    // Make sure things that come after fields are memory-aligned
    assert(sizeof(Fields) % sizeof(midx_t) == 0);

    midx_t *row_ptr = (midx_t *)(data_ptr + sizeof(Fields));
    midx_t *col_idx =
        (midx_t *)(((std::byte *)row_ptr) + ((height + 1) * sizeof(midx_t)));
    T *values = (T *)(((std::byte *)col_idx) + (non_zeros * sizeof(midx_t)));

    assert((midx_t)((std::byte *)row_ptr - data_ptr) == sizeof(Fields));
    assert((midx_t)((std::byte *)col_idx - (std::byte *)row_ptr) ==
           (height + 1) * sizeof(midx_t));
    assert((midx_t)((std::byte *)values - (std::byte *)col_idx) ==
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
      : CSRMatrix(get_cells<T>(file_path, tr, keep_rows, keep_cols), tr,
                  alloc) {}

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

    data = std::make_shared<std::vector<std::byte, Allocator>>(
        expected_data_size(), alloc);

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

  CSRMatrix(std::shared_ptr<std::vector<std::byte, Allocator>> &serialized_data,
            const Allocator &alloc = Allocator())
      : data(serialized_data), fields(utils::get_fields(serialized_data)),
        height(fields->height), width(fields->width),
        non_zeros(fields->non_zeros), transposed(fields->transposed) {
    auto [_row_ptr, _col_idx, _values] = get_offsets();
    row_ptr = _row_ptr;
    col_idx = _col_idx;
    values = _values;
  }

  // Copy constructor
  CSRMatrix(const CSRMatrix<T, Allocator> &mtx,
            const Allocator &alloc = Allocator())
      : data(std::make_shared<std::vector<std::byte, Allocator>>(
            *mtx.data.get(), alloc)) {
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
  std::shared_ptr<std::vector<std::byte, Allocator>> serialize() {
    return data;
  }
};

template <typename T = double> class BlockedCSRMatrix {
  using Allocator = utils::ContiguousAllocator<std::byte>;

private:
  std::shared_ptr<std::vector<std::byte, std::allocator<std::byte>>> data;
  BlockedFields *blocked_fields;

  size_t initial_data_size() { return sizeof(BlockedFields); }

  Allocator alloc;

  std::vector<CSRMatrix<T, Allocator>> csrs;
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
    std::byte *start = data->data();
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      blocked_fields->section_offst[i] = csrs[i].serialize()->data() - start;
      blocked_fields->section_start_row[i] = section_height * i;
    }
  }

  void compute_csrs_from_blocked_fields() {
    std::byte *start = data->data();
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
      : data(std::make_shared<std::vector<std::byte>>(initial_data_size())),
        blocked_fields(utils::get_blocked_fields(data)), alloc(data), csrs(),
        width(0), height(0), non_zeros(0) {
    static_assert(N_SECTIONS > 1);
    blocked_fields->n_sections = N_SECTIONS;

    // Compute how much space will be needed for the CSR representation
    auto fields = utils::read_fields(file_path, false, nullptr, keep_cols);
    auto expected_size = ((sizeof(T) + sizeof(midx_t)) * fields.non_zeros) +
                         // The + N_SECTIONS accounts for a wasted space in the
                         // row_ptr in every CSRMatrix
                         ((sizeof(midx_t)) * (fields.height + N_SECTIONS)) +
                         (sizeof(Fields) * N_SECTIONS);
    std::cout << "expected data size " << expected_size << std::endl;
    data->reserve(data->size() + expected_size);

    for (size_t i; i < blocked_fields->n_sections; ++i) {
      // TODO: compute keep_rows
      csrs.push_back(
          CSRMatrix<T, Allocator>(file_path, false, nullptr, keep_cols, alloc));
    }

    compute_class_fields();
    compute_blocked_fields_from_csrs();
  }

  BlockedCSRMatrix(std::shared_ptr<std::vector<std::byte>> serialized_data)
      : data(serialized_data), blocked_fields(utils::get_blocked_fields(data)),
        alloc(serialized_data), csrs(4, nullptr), width(0), height(0),
        non_zeros(0) {
    compute_csrs_from_blocked_fields();
    compute_class_fields();
  }

  ~BlockedCSRMatrix() {
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      csrs[i];
    }
  }

  SmallVec<T> row(midx_t i) {
    auto rel = i % N_SECTIONS;
    auto block_i = i - rel;
    return csrs[block_i]->row(rel);
  }

  CSRMatrix<T, Allocator> &block(midx_t i) {
    assert(i < N_SECTIONS);
    return csrs[i];
  }

  BlockedCSRMatrix filter(std::bitset<N_SECTIONS> bitmap) {
    auto new_data =
        std::make_shared<std::vector<std::byte>>(initial_data_size());
    auto bf = utils::get_blocked_fields(data);
    bf->n_sections = bitmap.count();

    auto new_alloc = utils::ContiguousAllocator(new_data);

    size_t j = 0;
    size_t section_height = height / blocked_fields->n_sections;
    for (size_t i; i < blocked_fields->n_sections; ++i) {
      if (!bitmap[i])
        continue;

      auto m = CSRMatrix<T, Allocator>(csrs[i], new_alloc);
      bf->section_offst[j] = new_data->size();
      bf->section_start_row[j] = blocked_fields->section_start_row[i];
    }

    return BlockedCSRMatrix(new_data);
  }

  // DO NOT WRITE TO THE OUTPUT OF THIS
  std::shared_ptr<std::vector<std::byte>> serialize() { return data; }
};

} // namespace matrix
