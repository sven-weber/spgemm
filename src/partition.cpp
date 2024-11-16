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
