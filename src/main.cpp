#include "bitmap.hpp"
#include "communication.hpp"
#include "matrix.hpp"
#include "matrix_impl.hpp"
#include "measure.hpp"
#include "mults.hpp"
#include "partition.hpp"
#include "parts.hpp"
#include "utils.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
  matrix::BlockedCSRMatrix<> m("matrices/test/A.mtx");
  for (size_t i = 0; i < N_SECTIONS; ++i) {
    auto b = m.block(i);
    utils::visualize(b.get(), std::format(" m.b{}", i));
  }

  /*auto ser = m.serialize();*/
  /*auto mm = matrix::BlockedCSRMatrix(ser);*/
  /*for (size_t i = 0; i < N_SECTIONS; ++i) {*/
  /*  auto b = mm.block(i);*/
  /*  utils::visualize(b.get(), std::format("mm.b{}", i));*/
  /*}*/
}
