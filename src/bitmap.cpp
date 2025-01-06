#include "bitmap.hpp"
#include "measure.hpp"
#include <bitset>
#include <iostream>
#include <omp.h>

#pragma omp declare reduction(or : std::bitset<N_SECTIONS> : omp_out |=        \
                                  omp_in)                                      \
    initializer(omp_priv = std::bitset<N_SECTIONS>())

namespace bitmap {
std::bitset<N_SECTIONS> compute_bitmap(matrix::CSRMatrix<> mat) {
  measure_point(measure::bitmaps, measure::MeasurementEvent::START);
  int section_width = mat.width / N_SECTIONS;
  assert(section_width > 0);

  auto map = std::bitset<N_SECTIONS>();
#pragma omp parallel for reduction(or : map)
  for (size_t row = 0; row < mat.height; row++) {
    auto [row_data, row_pos, row_len] = mat.row(row);
    for (size_t index = 0; index < row_len; index++) {
      map.set(std::min(row_pos[index] / section_width, (midx_t)N_SECTIONS - 1),
              true);
    }
  }

  std::cout << "Bitmap drop percentage: "
            << (N_SECTIONS - map.count()) / (double)N_SECTIONS << std::endl;

#ifndef NDEBUG
  std::cout << "Matrix size: " << mat.width << "-" << mat.height << std::endl;
  std::cout << "Section width: " << section_width << std::endl;
  std::cout << "bitmap.count = " << map.count() << std::endl;
  std::cout << "bitmap = " << map << std::endl;
#endif

  measure_point(measure::bitmaps, measure::MeasurementEvent::END);

  return map;
}
} // namespace bitmap
