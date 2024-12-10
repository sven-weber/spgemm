#pragma once

#include "matrix.hpp"
#include "measure.hpp"

#include <bitset>
#include <cassert>
#include <cstring>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <iostream>
#include <memory>
#include <mio/mmap.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace matrix {
namespace fmm = fast_matrix_market;

template <typename T> struct triplet_matrix {
  midx_t nrows, ncols;
  std::vector<midx_t> rows;
  std::vector<midx_t> cols;
  std::vector<T> vals;
};

namespace utils {

Fields read_fields(std::string file_path, bool transposed,
                   std::vector<midx_t> *keep_rows,
                   std::vector<midx_t> *keep_cols);

template <typename T> Fields *get_fields(T *serialized_data) {
  return static_cast<Fields *>(static_cast<void *>(serialized_data));
}

void write_matrix_market(std::string file_path, midx_t height, midx_t width,
                         std::vector<Cell<>> &lines);

template <typename T>
static inline void insertcpy(std::vector<T> &dst, T *src, midx_t amt) {
  dst.insert(dst.end(), src, src + amt);
}

BlockedFields *get_blocked_fields(std::byte *serialized_data);

} // namespace utils

template <typename T = double> class Cells {
public:
  // Don't use!
  midx_t height;
  midx_t width;
  std::vector<std::unordered_map<midx_t, midx_t>> _pos;
  std::vector<std::unordered_map<midx_t, T>> _cells;

public:
  // Takes in the number of rows
  Cells(midx_t height, midx_t width, midx_t non_zeros = 0)
      : height(height), width(width), _cells(height), _pos(height){}
  ~Cells() = default;

  size_t expected_data_size() const {
    return sizeof(Fields) + ((height + 1) * sizeof(midx_t)) +
           (non_zeros() * sizeof(midx_t)) + (non_zeros() * sizeof(T));
  }

  // Add one cell to the list of Cells
  // usage: add({row, col}, val)
  // indexes are 0-based
  // If a cell in {row, col} has already been inserted, the value is summed
  // to the previous
  // Concurrent accesses to different positions are supported but not on the
  // second one!
  void add(CellPos pos, T val) {
    assert(pos.first < _cells.size());
    assert(val != 0);

    if (!_cells[pos.first].contains(pos.second)) {
      /*++_non_zeros;*/
      _pos[pos.first].insert({pos.second, cells_in_row(pos.first)});
      _cells[pos.first].insert({pos.second, val});
    } else {
      _cells[pos.first][pos.second] += val;
      if (_cells[pos.first][pos.second] == 0) {
        _cells[pos.first].erase(pos.second);
        _pos[pos.first].erase(pos.second);
      }
    }
  }

  // Returns the amount of cells (i.e., non_zeros)
  midx_t non_zeros() const {
    auto non_zeros = 0;
    for (auto m : _cells)
      non_zeros += m.size();
    return non_zeros;
  }

  // Returns the number of cells in a row
  midx_t cells_in_row(midx_t row) const {
    assert(row < _cells.size());
    return _cells[row].size();
  }
};

template <typename Iterator> class IteratorStreambuf : public std::streambuf {
public:
  IteratorStreambuf(Iterator begin, Iterator end) : current(begin), last(end) {}

protected:
  // Called when more characters are needed
  int_type underflow() override {
    if (current == last) {
      return traits_type::eof(); // End of input
    }
    return traits_type::to_int_type(*current);
  }

  // Called when characters are consumed (reading or extracting)
  int_type uflow() override {
    if (current == last) {
      return traits_type::eof(); // End of input
    }
    return traits_type::to_int_type(*current++);
  }

  // Called for unget (putting back a character)
  int_type pbackfail(int_type ch) override {
    if (current == last || ch != traits_type::to_int_type(*(--current))) {
      return traits_type::eof(); // No character to unget
    }
    return ch;
  }

  std::streamsize showmanyc() override {
    return std::distance(current, last); // Number of characters left
  }

private:
  Iterator current, last;
};

template <typename Iterator> class IteratorInputStream : public std::istream {
public:
  IteratorInputStream(Iterator begin, Iterator end)
      : std::istream(&buffer), buffer(begin, end) {}

private:
  IteratorStreambuf<Iterator> buffer;
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

  measure_point(measure::read_triplets, measure::MeasurementEvent::START);
  auto fields = utils::read_fields(file_path, transposed, keep_rows, keep_cols);

  std::error_code error;
  mio::mmap_sink ro_mmap = mio::make_mmap_sink(file_path, error);
  assert(!error);
  IteratorInputStream stream(ro_mmap.begin(), ro_mmap.end());

  auto cells = full ? Cells<T>(fields.height, fields.width, fields.non_zeros)
                    : Cells<T>(fields.height, fields.width);
  // cells._pos.resize(fields.height);

  triplet_matrix<T> tm;
  fmm::read_matrix_market_triplet(stream, tm.nrows, tm.ncols, tm.rows, tm.cols,
                                  tm.vals);
  measure_point(measure::read_triplets, measure::MeasurementEvent::END);

  std::cout << "LIBRARY LOAD FINISHED; MATRIX TRANSFORMATION" << std::endl;

  measure_point(measure::triplets_to_map, measure::MeasurementEvent::START);

  const constexpr uint32_t ROW_MUTEX_COUNT = 10000;
  std::mutex mutexes[ROW_MUTEX_COUNT];
#pragma omp parallel for
  for (midx_t i = 0; i < tm.vals.size(); ++i) {
    auto _row = tm.rows[i];
    auto _col = tm.cols[i];
    auto val = tm.vals[i];

    auto row = transposed ? _col : _row;
    auto col = transposed ? _row : _col;

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

    std::lock_guard<std::mutex> guard(mutexes[row % ROW_MUTEX_COUNT]);
    cells._pos[row].insert({col, cells.cells_in_row(row)});
    cells._cells[row].insert({col, val});
  }
  measure_point(measure::triplets_to_map, measure::MeasurementEvent::END);

  ro_mmap.unmap();
  return cells;
}

using Data = std::pair<std::byte *, size_t>;

template <typename T = double> class CSRMatrix {
protected:
  std::byte *data;
  size_t size;

private:
  Fields *fields;

  size_t expected_data_size() {
    return sizeof(Fields) + ((height + 1) * sizeof(midx_t)) +
           (non_zeros * sizeof(midx_t)) + (non_zeros * sizeof(T));
  }

  // Returns the expected size of data in bytes
  std::tuple<midx_t *, midx_t *, T *> get_offsets() {
    assert(size == expected_data_size());
    std::byte *data_ptr = data;

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
  midx_t height = 0;
  midx_t width = 0;
  midx_t non_zeros = 0;
  bool transposed = false;

  midx_t *row_ptr = nullptr;
  midx_t *col_idx = nullptr;
  T *values = nullptr;

  CSRMatrix(const Cells<T> &cells, const Data &d, bool tr = false)
      : data(std::get<0>(d)), size(std::get<1>(d)),
        fields(utils::get_fields(data)), height(cells.height),
        width(cells.width), non_zeros(cells.non_zeros()), transposed(tr) {
#ifndef NDEBUG
    for (int row = 0; row < cells._cells.size(); row++) {
      for (auto [col, val] : cells._cells[row]) {
        assert(row < height);
        assert(col < width);
        assert(val != 0);
      }
    }
#endif

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
    measure_point(measure::build_csr, measure::MeasurementEvent::START);
    #pragma omp parallel for
    for (int row = 0; row < cells._cells.size(); row++) {
      midx_t max_pos = 0;
      for (auto [col, val] : cells._cells[row]) {
        auto index = row_ptr[row] + cells._pos[row].at(col);
        col_idx[index] = col;
        values[index] = val;
      }
    }
    measure_point(measure::build_csr, measure::MeasurementEvent::END);
  }

  CSRMatrix(const Data &d)
      : data(std::get<0>(d)), size(std::get<1>(d)),
        fields(utils::get_fields(data)), height(fields->height),
        width(fields->width), non_zeros(fields->non_zeros),
        transposed(fields->transposed) {
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
  Data serialize() { return {data, size}; }
};

template <typename T = double> class ManagedCSRMatrix : public CSRMatrix<T> {
private:
  Data alloc_for_cells(const Cells<T> &cells) {
    auto sz = cells.expected_data_size();
    auto vec = new std::byte[sz];
    return {vec, sz};
  }

public:
  ManagedCSRMatrix(std::string file_path, bool tr = false,
                   std::vector<midx_t> *keep_rows = nullptr,
                   std::vector<midx_t> *keep_cols = nullptr)
      : ManagedCSRMatrix<T>(get_cells<T>(file_path, tr, keep_rows, keep_cols),
                            tr) {}

  ManagedCSRMatrix(const Cells<T> &cells, bool tr = false)
      : CSRMatrix<T>(cells, alloc_for_cells(cells), tr) {}

  ~ManagedCSRMatrix() { delete[] CSRMatrix<T>::data; }
};

#define section_height(height) ((height) / N_SECTIONS)

template <typename T = double> class BlockedCSRMatrix {
protected:
  std::byte *data;
  size_t size;

  static size_t initial_data_size() { return sizeof(BlockedFields); }

private:
  BlockedFields *blocked_fields;

  std::vector<std::shared_ptr<CSRMatrix<T>>> csrs;
  std::unordered_map<midx_t, std::shared_ptr<CSRMatrix<T>>> start_row_to_csrs;

  void compute_class_fields() {
    width = csrs[0]->width;
    height = blocked_fields->height;

#ifndef NDEBUG
    for (auto csr : csrs) {
      assert(csr->width == width);
    }
#endif

    for (size_t i = 0; i < blocked_fields->n_sections; ++i) {
      start_row_to_csrs[blocked_fields->section_start_row[i]] = csrs[i];
      non_zeros += csrs[i]->non_zeros;
    }
  }

  void compute_blocked_fields_from_csrs() {
    // Here we assume all sections are filled
    size_t sh = section_height(height);
    std::byte *start = data;
    for (size_t i = 0; i < blocked_fields->n_sections; ++i) {
      blocked_fields->section_offst[i] =
          std::get<0>(csrs[i]->serialize()) - start;
      blocked_fields->section_start_row[i] = sh * i;
    }
  }

  void compute_csrs_from_blocked_fields() {
    for (size_t i = 0; i < blocked_fields->n_sections; ++i) {
      auto begin = blocked_fields->section_offst[i];
      auto end = (i == blocked_fields->n_sections - 1)
                     ? size
                     : blocked_fields->section_offst[i + 1];
      auto sz = end - begin;

      Data d = {data + begin, sz};
      csrs.push_back(std::make_shared<CSRMatrix<T>>(d));
    }
  }

public:
  midx_t height = 0;
  midx_t width = 0;
  midx_t non_zeros = 0;

  BlockedCSRMatrix(std::tuple<Data, std::vector<Cells<T>>, Fields> d)
      : data(std::get<0>(std::get<0>(d))), size(std::get<1>(std::get<0>(d))) {
    auto cells = std::get<1>(d);
    auto fields = std::get<2>(d);
    csrs = std::vector<std::shared_ptr<CSRMatrix<T>>>();

    blocked_fields = utils::get_blocked_fields(data);
    blocked_fields->n_sections = N_SECTIONS;
    blocked_fields->height = fields.height;

    std::cout << "Make csrs" << std::endl;
    auto partition_height = section_height(fields.height);
    auto idx = initial_data_size();
    for (size_t i = 0; i < N_SECTIONS; ++i) {
      auto cell = cells[i];
      auto row = i * partition_height;

      auto sz = cell.expected_data_size();
      Data d = {data + idx, sz};
      idx += sz;

      blocked_fields->section_start_row[i] = row;
      blocked_fields->section_offst[i] = idx;

      csrs.push_back(std::make_shared<CSRMatrix<T>>(cell, d, false));
    }

    compute_class_fields();
    compute_blocked_fields_from_csrs();

    assert(blocked_fields->n_sections == N_SECTIONS);
    assert(blocked_fields->height == height);
  }

  BlockedCSRMatrix(const Data &d)
      : data(std::get<0>(d)), size(std::get<1>(d)),
        blocked_fields(utils::get_blocked_fields(data)) {
    compute_csrs_from_blocked_fields();
    compute_class_fields();
  }

  SmallVec<T> row(midx_t i) {
    auto block_i = std::min(i / section_height(height), (midx_t)N_SECTIONS - 1);
    auto start_row = block_i * section_height(height);
    auto rel = i - start_row;

    return start_row_to_csrs.at(start_row)->row(rel);
  }

  std::shared_ptr<CSRMatrix<T>> block(midx_t i) {
    auto block_i = std::min(i / section_height(height), (midx_t)N_SECTIONS - 1);
    auto start_row = block_i * section_height(height);

    return start_row_to_csrs.at(start_row);
  }

  std::tuple<std::shared_ptr<CSRMatrix<T>>, midx_t> block_i(midx_t i) {
    assert(i < blocked_fields->n_sections);
    return {csrs[i], blocked_fields->section_start_row[i]};
  }

  std::vector<std::byte> filter(std::bitset<N_SECTIONS> bitmap,
                                size_t size = 0) {
    auto new_data = std::vector<std::byte>(initial_data_size());
    new_data.reserve(size != 0 ? size : initial_data_size());

    BlockedFields bf;
    bf.n_sections = bitmap.count();
    bf.height = height;

    size_t j = 0;
    size_t sz = initial_data_size();
    for (size_t i = 0; i < blocked_fields->n_sections; ++i) {
      if (!bitmap[i])
        continue;

      auto [csr_data, sz_] = csrs[i]->serialize();
      new_data.insert(new_data.end(), csr_data, csr_data + sz_);

      bf.section_offst[j] = sz;
      bf.section_start_row[j] = blocked_fields->section_start_row[i];

      sz += sz_;
      ++j;
    }
    std::memcpy(new_data.data(), &bf, sizeof(bf));

    return new_data;
  }

  // DO NOT WRITE TO THE OUTPUT OF THIS
  Data serialize() { return {data, size}; }
};

template <typename T = double>
class ManagedBlockedCSRMatrix : public BlockedCSRMatrix<T> {
private:
  static std::pair<std::vector<Cells<T>>, size_t>
  get_cells_sections(std::string file_path, bool transposed,
                     const std::vector<std::vector<midx_t>> &keep_rows_sections,
                     std::vector<midx_t> *keep_cols) {
    auto keep_rows_map =
        std::unordered_map<midx_t, std::tuple<midx_t, size_t>>();
    for (size_t s = 0; s < N_SECTIONS; ++s) {
      auto keep_rows = keep_rows_sections[s];
      for (midx_t i = 0; i < keep_rows.size(); ++i) {
        keep_rows_map.insert({keep_rows[i], {i, s}});
      }
    }

    auto keep_cols_map = std::unordered_map<midx_t, midx_t>();
    if (keep_cols != nullptr)
      for (midx_t i = 0; i < keep_cols->size(); ++i) {
        keep_cols_map.insert({(*keep_cols)[i], i});
      }

    measure_point(measure::read_triplets, measure::MeasurementEvent::START);
    auto fields = utils::read_fields(file_path, transposed, nullptr, keep_cols);
    std::error_code error;
    mio::mmap_sink ro_mmap = mio::make_mmap_sink(file_path, error);
    assert(!error);
    IteratorInputStream stream(ro_mmap.begin(), ro_mmap.end());

    triplet_matrix<T> tm;
    fmm::read_matrix_market_triplet(stream, tm.nrows, tm.ncols, tm.rows,
                                    tm.cols, tm.vals);
    measure_point(measure::read_triplets, measure::MeasurementEvent::END);

    std::vector<Cells<T>> cells_sections;
    for (size_t s = 0; s < N_SECTIONS; ++s) {
      auto cells = Cells<T>(keep_rows_sections[s].size(), fields.width);
      cells_sections.push_back(cells);
    }

    measure_point(measure::triplets_to_map, measure::MeasurementEvent::START);
    auto max_section_rows = (tm.nrows / N_SECTIONS) + tm.nrows % N_SECTIONS;
    std::mutex mutexes[N_SECTIONS][max_section_rows];

#pragma omp parallel for
    for (midx_t i = 0; i < tm.vals.size(); ++i) {
      auto _row = tm.rows[i];
      auto _col = tm.cols[i];
      auto val = tm.vals[i];

      auto row = transposed ? _col : _row;
      auto col = transposed ? _row : _col;

      if (!keep_rows_map.contains(row))
        continue;
      auto [mapped_row, sec] = keep_rows_map[row];

      if (keep_cols != nullptr) {
        if (!keep_cols_map.contains(col))
          continue;

        col = keep_cols_map[col];
      }

      assert(mapped_row < max_section_rows);
      std::lock_guard<std::mutex> guard(mutexes[sec][mapped_row]);
      cells_sections[sec]._pos[mapped_row].insert({col, cells_sections[sec].cells_in_row(mapped_row)});
      cells_sections[sec]._cells[mapped_row].insert({col, val});
    }
    measure_point(measure::triplets_to_map, measure::MeasurementEvent::END);

    ro_mmap.unmap();

    size_t exp_size = 0;
    for (size_t s = 0; s < N_SECTIONS; ++s) {
      exp_size += cells_sections[s].expected_data_size();
    }
    return {cells_sections, exp_size};
  }

  static std::tuple<Data, std::vector<Cells<T>>, Fields>
  alloc_for_cells(std::string file_path,
                  std::vector<midx_t> *keep_cols = nullptr) {
    auto fields = utils::read_fields(file_path, false, nullptr, keep_cols);

    static_assert(N_SECTIONS > 1);

    auto partition_height = section_height(fields.height);
    std::vector<std::vector<midx_t>> keep_rows_sections;
    for (size_t i = 0; i < N_SECTIONS; ++i) {
      std::vector<midx_t> keep_rows;
      auto start = i * partition_height;
      auto end =
          (i == (N_SECTIONS - 1)) ? fields.height : (i + 1) * partition_height;
      for (size_t j = start; j < end; ++j) {
        keep_rows.push_back(j);
      }
      keep_rows_sections.push_back(keep_rows);
    }

    auto [cells, sz] =
        get_cells_sections(file_path, false, keep_rows_sections, keep_cols);
    sz += BlockedCSRMatrix<T>::initial_data_size();

    auto vec = new std::byte[sz];
    Data d = std::make_pair(vec, sz);
    return std::make_tuple(d, cells, fields);
  }

public:
  ManagedBlockedCSRMatrix(std::string file_path,
                          std::vector<midx_t> *keep_cols = nullptr)
      : BlockedCSRMatrix<T>(alloc_for_cells(file_path, keep_cols)) {}

  /*~ManagedBlockedCSRMatrix() { delete[] BlockedCSRMatrix<T>::data; }*/
};

} // namespace matrix
