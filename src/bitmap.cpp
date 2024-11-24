#include "bitmap.hpp"
#include <bitset>
#include <iostream>

namespace bitmap {
std::bitset<N_SECTIONS> compute_bitmap(matrix::CSRMatrix<> mat) {
  std::cout << "Computing bitmap" << std::endl;
  int section_width = mat.width / N_SECTIONS;

  auto map = std::bitset<N_SECTIONS>();
  for (size_t row = 0; row < mat.height; row++) {
    auto [row_data, row_pos, row_len] = mat.row(row);
    for (size_t index = 0; index < row_len; index++) {
      map.set(std::min(row_pos[index] / section_width, (midx_t) N_SECTIONS-1), true);
    }
  }

  std::cout << "Matrix size: " << mat.width << "-" << mat.height << std::endl;
  std::cout << "Section width: " << section_width << std::endl;
  std::cout << "Drop percentage: " << (N_SECTIONS - map.count())/(double)N_SECTIONS << std::endl;
  return map;
}
} // namespace bitmap
