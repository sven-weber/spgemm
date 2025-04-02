#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/SpMat.h"
#include "CombBLAS/SpParMat1D.h"
#include "communication.hpp"
#include "mpi.h"
#include "mults.hpp"
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace mults {
shared_ptr<combblas::CommGrid> start1DWorld() {
  shared_ptr<combblas::CommGrid> fullWorld;
  fullWorld.reset(new combblas::CommGrid(MPI_COMM_WORLD, 0, 0));
  return fullWorld;
}

Sp1D loadMatrix(std::string path_A, shared_ptr<combblas::CommGrid> fullWorld) {
  combblas::SpParMat<int64_t, double, combblas::SpDCCols<int64_t, double>>
      Readingmatrix(fullWorld);

  Readingmatrix.ParallelReadMM(path_A, false, combblas::maximum<double>());
  typedef combblas::PlusTimesSRing<double, double> PTFF;
  Sp2D A2D(Readingmatrix);
  return Sp1D(A2D, combblas::SpParMat1DTYPE::COLWISE);
}

CombBLASMatrixMultiplication::CombBLASMatrixMultiplication(
    int rank, int n_nodes, partition::Partitions partitions, std::string path_A)
    : MatrixMultiplication(rank, n_nodes, partitions), fullWorld(start1DWorld()),
      A1D(loadMatrix(path_A, fullWorld)), B1D(loadMatrix(path_A, fullWorld)),
      C1D(A1D) {}

void CombBLASMatrixMultiplication::save_result(std::string path) {
  Sp2D C2DFrom1D(C1D);
  C2DFrom1D.ParallelWriteMM(path, false);
}

void CombBLASMatrixMultiplication::gemm(
    std::vector<size_t> serialized_sizes_B_bytes, size_t max_size_B_bytes) {
  C1D = std::move(
      combblas::Mult_AnXBn_1D<PTFF, double, combblas::SpDCCols<int64_t, double>,
                              int64_t, double, double,
                              combblas::SpDCCols<int64_t, double>,
                              combblas::SpDCCols<int64_t, double>>(A1D, B1D));
}

} // namespace mults
