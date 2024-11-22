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
  matrix::BlockedCSRMatrix<> m("matrices/test/B.mtx");
  auto b0 = m.block(0);
  utils::visualize(b0, "b0");
}
