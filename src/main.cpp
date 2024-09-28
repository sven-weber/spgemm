#include "matrix.hpp"
#include <iostream>

void visualize(matrix::CSRMatrix &csr) {
  auto matrix = (double *)calloc(csr.height * csr.width, sizeof(double));

  for (size_t i = 0; i < csr.height; i++) {
    auto pos = csr.row_ptr[i];
    auto end = csr.row_ptr[i + 1];

    while (pos < end) {
      auto j = csr.col_idx[pos];
      auto val = csr.values[pos];
      matrix[(i * csr.width) + j] = val;
      ++pos;
    }
  }

  for (size_t i = 0; i < csr.height; i++) {
    std::cout << i + 1 << ":\t";
    for (size_t j = 0; j < csr.width; j++) {
      std::cout << matrix[(i * csr.width) + j] << "\t";
    }
    std::cout << "\n";
  }
}

int main() {
  matrix::CSRMatrix mm("matrices/small2/A.mtx");
  visualize(mm);
}
