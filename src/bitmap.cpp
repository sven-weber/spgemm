#include "bitmap.hpp"
#include <bitset>
#include <cmath>
#include <iostream>

namespace bitmap {
std::bitset<N_SECTIONS> compute_bitmap(matrix::CSRMatrix<> mat) {
  int section_width = ceil((float)mat.width / (float)N_SECTIONS);

  auto map = std::bitset<N_SECTIONS>();
  for (size_t row = 0; row < mat.height; row++) {
    auto [row_data, row_pos, row_len] = mat.row(row);
    for (size_t index = 0; index < row_len; index++) {
      map[row_pos[index] / section_width] = true;
    }
  }

  std::cout << "Matrix size: " << mat.width << "-" << mat.height << std::endl;
  std::cout << "Section width: " << section_width << std::endl;
  std::cout << "Number of false bits: " << map.count() << std::endl;
  return map;
}
} // namespace bitmap
