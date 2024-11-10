#pragma once
#include "matrix.hpp"

namespace bitmap {
  extern int n_sections;
  typedef std::pair<int, int> section; 

  std::vector<section> compute_drop_sections(matrix::CSRMatrix mat);
}