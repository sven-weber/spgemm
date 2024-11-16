#pragma once
#include "matrix.hpp"
#include <bitset>

namespace bitmap {
std::bitset<N_SECTIONS> compute_bitmap(matrix::CSRMatrix<> mat);
} // namespace bitmap
