#include <algorithm>
#include <cassert>
#include <execution>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <tbb/parallel_sort.h>

#include "partition.hpp"
#include "measure.hpp"

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
  std::iota(indices.begin(), indices.end(), 0);

  tbb::parallel_sort(indices.begin(), indices.end(),
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
  std::iota(indices.begin(), indices.end(), 0);

  tbb::parallel_sort(indices.begin(), indices.end(),
                     [&](int a, int b) { return values[a] > values[b]; });
  return indices;
}

template <typename T>
std::vector<std::pair<float, midx_t>>
calculate_avg_indices_col(matrix::CSRMatrix<T> matrix, Shuffle *shuffled_rows,
                          Shuffle *shuffled_cols) {
  std::vector<std::pair<float, midx_t>> indices(matrix.height);

#pragma omp parallel
  for (size_t col = 0; col < indices.size(); col++) {
    auto [col_data, col_pos, col_len] = matrix.col(col);
    float avg = 0.0;
    if (col_len == 0) {
      avg = matrix.height;
    } else {
      for (size_t row = 0; row < col_len; row++) {
        avg += (*shuffled_rows)[col_pos[row]];
      }
      avg /= col_len;
    }
    indices[col] = {avg, col};
  }
  tbb::parallel_sort(indices.begin(), indices.end(),
                     [&](auto a, auto b) { return a.first < b.first; });
  return indices;
}

template <typename T>
std::vector<std::pair<float, midx_t>>
calculate_avg_indices_row(matrix::CSRMatrix<T> matrix, Shuffle *shuffled_rows,
                          Shuffle *shuffled_cols) {
  std::vector<std::pair<float, midx_t>> indices(matrix.height);

#pragma omp parallel
  for (size_t row = 0; row < indices.size(); row++) {
    auto [row_data, row_pos, row_len] = matrix.row(row);
    float avg = 0.0;
    if (row_len == 0) {
      avg = matrix.width;
    } else {
      for (size_t col = 0; col < row_len; col++) {
        avg += (*shuffled_cols)[row_pos[col]];
      }
      avg /= row_len;
    }
    indices[row] = {avg, row};
  }
  tbb::parallel_sort(indices.begin(), indices.end(),
                     [&](auto a, auto b) { return a.first < b.first; });
  return indices;
}

void iterative_shuffle(std::string C_sparsity_path,
                       std::vector<midx_t> *shuffled_rows,
                       std::vector<midx_t> *shuffled_cols) {
  std::iota(shuffled_rows->begin(), shuffled_rows->end(), 0);
  std::iota(shuffled_cols->begin(), shuffled_cols->end(), 0);

  matrix::ManagedCSRMatrix<short> C_transposed(C_sparsity_path, true, nullptr,
                                          nullptr);
  matrix::ManagedCSRMatrix<short> C(C_sparsity_path, false, nullptr, nullptr);

  std::cout << "MATRIX LOADED; SHUFFLING BEGINS;" << std::endl;
  
  // Dont include the loading time in the shuffling timing!
  measure_point(measure::shuffle, measure::MeasurementEvent::START);

  float variance = 0;
  float stopping_variance = -1;
  double start = omp_get_wtime();
  // We do shuffling for a maximus of 30 seconds
  double stopping_time = std::min(30.0, shuffled_rows->size() * 0.00025);
  int i = 0;

  while (i < 4) {
    float sum_x = 0;
    float sum_x2 = 0;
    bool transpose = i % 2 != 0;

    if (transpose) {
      std::vector<std::pair<float, midx_t>> avg_indices =
          calculate_avg_indices_col(C_transposed, shuffled_rows, shuffled_cols);

      Shuffle tmp_shuffled_cols(shuffled_cols->size());

#pragma omp parallel for
      for (int i = 0; i < shuffled_cols->size(); i++) {
        tmp_shuffled_cols[avg_indices[i].second] = i;
      }

      (*shuffled_cols) = tmp_shuffled_cols;
    } else {
      std::vector<std::pair<float, midx_t>> avg_indices =
          calculate_avg_indices_row(C, shuffled_rows, shuffled_cols);

      Shuffle tmp_shuffled_rows(shuffled_rows->size());

#pragma omp parallel for reduction(+ : sum_x, sum_x2)
      for (int i = 0; i < shuffled_rows->size(); i++) {
        float x = avg_indices[i].first - i;
        sum_x += x / shuffled_rows->size();
        sum_x2 += (x * x) / shuffled_rows->size();
        tmp_shuffled_rows[avg_indices[i].second] = i;
      }

      variance = (sum_x2 - (sum_x * sum_x)) *
                 (shuffled_rows->size() / (shuffled_rows->size() - 1));

      std::cout << "Shuffling iteration " << i << ": " << variance << std::endl;

      (*shuffled_rows) = tmp_shuffled_rows;
    }
    i++;
  }

  Shuffle tmp_shuffled_rows(shuffled_rows->size());
#pragma omp parallel for
  for (int i = 0; i < shuffled_rows->size(); i++) {
    tmp_shuffled_rows[(*shuffled_rows)[i]] = i;
  }
  (*shuffled_rows) = tmp_shuffled_rows;

  Shuffle tmp_shuffled_cols(shuffled_cols->size());
#pragma omp parallel for
  for (int i = 0; i < shuffled_cols->size(); i++) {
    tmp_shuffled_cols[(*shuffled_cols)[i]] = i;
  }
  (*shuffled_cols) = tmp_shuffled_cols;
  
  measure_point(measure::shuffle, measure::MeasurementEvent::END);
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

bool load_shuffle(std::string file, Shuffle &shuffle) {
  std::error_code error;
  mio::mmap_sink ro_mmap = mio::make_mmap_sink(file, error);
  matrix::IteratorInputStream map(ro_mmap.begin(), ro_mmap.end());
  if (!error) {
    assert(!map.fail());

    std::string line;
    int index = 0;
    while (std::getline(map, line)) {
      try {
        shuffle[index] = std::stoi(line);
        index++;
      } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid number in file: " << line << std::endl;
      }
    }

    ro_mmap.unmap();
    return true;
  } else {
    return false;
  }
}

} // namespace partition
