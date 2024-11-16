#include "bitmap.hpp"
#include <cmath>
#include <vector>

namespace bitmap {
int n_sections;

std::vector<matrix::section> compute_drop_sections(matrix::CSRMatrix<> mat) {
  auto drop_sections = std::vector<matrix::section>();
  int section_width = ceil((float)mat.width / (float)n_sections);

  auto map = std::vector<bool>(n_sections, false);
  for (size_t row = 0; row < mat.height; row++) {
    auto [row_data, row_pos, row_len] = mat.row(row);
    for (size_t index = 0; index < row_len; index++) {
      map[row_pos[index] / section_width] = true;
    }
  }

  int start = -1;
  bool in_section = false;
  for (size_t i = 0; i < n_sections; i++) {
    if (!map[i]) {
      if (!in_section) {
        start = i * section_width;
        in_section = true;
      }
    } else if (in_section) {
      in_section = false;
      drop_sections.push_back({start, i * section_width});
    }
  }

  if (in_section) {
    drop_sections.push_back({start, mat.width});
  }

  return drop_sections;
}
} // namespace bitmap
