#pragma once
#include "matrix.hpp"

namespace bitmap {
extern int n_sections;

std::vector<matrix::section> compute_drop_sections(matrix::CSRMatrix mat);
} // namespace bitmap
