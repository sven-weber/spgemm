#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>

#include "partition.hpp"

namespace partition {

Shuffle shuffle(size_t size) {
  auto shuffled = Shuffle(size);

  for (size_t i = 0; i < size; i++) {
    shuffled[i] = i;
  }

#ifndef NSHUFFLE
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffled.begin(), shuffled.end(), g);
#endif

#ifndef NDEBUG
  std::cout << "Shuffled indices: ";
  for (size_t i = 0; i < size; i++) {
    std::cout << shuffled[i] << " ";
  }
  std::cout << std::endl;
#endif

  return shuffled;
}

Shuffle shuffle_avg(matrix::CSRMatrix<> matrix) {
  std::vector<double> values(matrix.height);
  for (size_t row = 0; row < matrix.height; row++) {
    auto [row_data, row_pos, row_len] = matrix.row(row);
    double avg = 0;
    for (size_t col = 0; col < row_len; col++) {
      avg += row_pos[col] / row_len;
    }
    values[row] = avg;
  }

  std::vector<midx_t> indices(values.size());
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return values[a] > values[b]; });
  return indices;
}

Shuffle shuffle_min(matrix::CSRMatrix<> matrix) {
  std::vector<double> values(matrix.height);
  for (size_t row = 0; row < matrix.height; row++) {
    auto [row_data, row_pos, row_len] = matrix.row(row);
    values[row] = row_len > 0 ? row_pos[0] : matrix.width;
  }

  std::vector<midx_t> indices(values.size());
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return values[a] > values[b]; });
  return indices;
}

std::vector<std::pair<float, midx_t>> calculate_avg_indices_col(matrix::CSRMatrix<> matrix, Shuffle *shuffled_cols) {
  std::vector<std::pair<float, midx_t>> indices(matrix.width);
  for (size_t col = 0; col < indices.size(); col++) {
    auto [col_data, col_pos, col_len] = matrix.col(col);
    float avg = 0.0;
    if (col_len == 0) {
      avg = matrix.width;
    } else {
      for (size_t row = 0; row < col_len; row++) {
        avg += col_pos[row];
      }
      avg /= col_len;
    }
    indices[col] = {avg, col};
  }
  std::sort(indices.begin(), indices.end(),
            [&](auto a, auto b) { return a.first < b.first; });
  return indices;
}

std::vector<std::pair<float, midx_t>> calculate_avg_indices_row(matrix::CSRMatrix<> matrix, Shuffle *shuffled_rows) {
  std::vector<std::pair<float, midx_t>> indices(matrix.height);
  for (size_t row = 0; row < indices.size(); row++) {
    auto [row_data, row_pos, row_len] = matrix.row(row);
    float avg = 0.0;
    if (row_len == 0) {
      avg = matrix.width;
    } else {
      for (size_t col = 0; col < row_len; col++) {
        avg += row_pos[col];
      }
      avg /= row_len;
    }
    indices[row] = {avg, row};
  }
  std::sort(indices.begin(), indices.end(),
            [&](auto a, auto b) { return a.first < b.first; });
  return indices;
}
  

std::pair<Shuffle, Shuffle> iterative_shuffle(matrix::CSRMatrix<> C, std::string C_sparsity_path, const int iterations) {
  Shuffle shuffled_rows(C.height);
  Shuffle shuffled_cols(C.width);

  std::iota(shuffled_rows.begin(), shuffled_rows.end(), 0);
  std::iota(shuffled_cols.begin(), shuffled_cols.end(), 0);
  
  for (int i = 0; i < iterations; i++) {
    bool transpose = i % 2 != 0;

    if (transpose) {
      matrix::ManagedCSRMatrix<> mat(C_sparsity_path, transpose, &shuffled_cols, &shuffled_rows);
      std::vector<std::pair<float, midx_t>> avg_indices = calculate_avg_indices_col(mat, &shuffled_cols);

      Shuffle tmp_shuffled_cols(shuffled_cols.size());
      for (int i = 0; i < shuffled_cols.size(); i++) {
        tmp_shuffled_cols[i] = shuffled_cols[avg_indices[i].second];
      }
      shuffled_cols = tmp_shuffled_cols;
    } else {
      matrix::ManagedCSRMatrix<> mat(C_sparsity_path, transpose, &shuffled_rows, &shuffled_cols);
      std::vector<std::pair<float, midx_t>> avg_indices = calculate_avg_indices_row(mat, &shuffled_rows);

      Shuffle tmp_shuffled_rows(shuffled_rows.size());
      for (int i = 0; i < shuffled_rows.size(); i++) {
        tmp_shuffled_rows[i] = shuffled_rows[avg_indices[i].second];
      }
      shuffled_rows = tmp_shuffled_rows;
    } 
  }

  return std::make_pair(shuffled_rows, shuffled_cols);
}

void save_partitions(Partitions &partitions, std::string file) {
  auto csv = std::ofstream(file);
  assert(!csv.fail());

  csv << "rank,start_row,end_row,start_col,end_col" << std::endl;

  for (size_t i = 0; i < partitions.size(); ++i) {
    auto part = partitions[i];
    csv << i << "," << part.start_row << "," << part.end_row << ","
        << part.start_col << "," << part.end_col << std::endl;
  }

  csv.close();
}

void save_shuffle(Shuffle &shuffle, std::string file) {
  auto map = std::ofstream(file);
  assert(!map.fail());

  for (auto i : shuffle)
    map << i << std::endl;

  map.close();
}

} // namespace partition
