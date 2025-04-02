#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpDCCols.h"
#include "Glue.h"
#include "CCGrid.h"
#include "Reductions.h"
#include "Multiplier.h"
#include "SplitMatDist.h"
#include "communication.hpp"
#include "mpi.h"
#include "mults.hpp"
#include "utils.hpp"
#include <cstring>
#include <cmath>
#include <iostream>
#include <unistd.h>

double comm_bcast;
double comm_reduce;
double comp_summa;
double comp_reduce;
double comp_result;
double comp_reduce_layer;
double comp_split;
double comp_trans;
double comm_split;

namespace mults {
World3D start3DWorld(unsigned C_FACTOR) {
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	unsigned GRCOLS = (int) sqrt((nprocs/C_FACTOR) + 0.5);

  combblas::CCGrid CMG(C_FACTOR, GRCOLS);
  shared_ptr<combblas::CommGrid> layerGrid;
  layerGrid.reset( new combblas::CommGrid(CMG.layerWorld, 0, 0) );
  combblas::FullyDistVec<int32_t, int32_t> p(layerGrid);
  return std::make_pair(CMG, p);
}

Sp3D loadMatrix(std::string path_A, bool rowsplit, combblas::CCGrid CMG, combblas::FullyDistVec<int32_t, int32_t> p) {
  Sp3D splitA;
  Sp3D *A = ReadMat<double>(path_A, CMG, true, p);
  SplitMat(CMG, A, splitA, rowsplit);
  return splitA;
}

CombBLAS3DMatrixMultiplication::CombBLAS3DMatrixMultiplication(
    int rank, int n_nodes, partition::Partitions partitions, std::string path_A, unsigned c)
    : MatrixMultiplication(rank, n_nodes, partitions), fullWorld(start3DWorld(c)),
      A3D(loadMatrix(path_A, false, fullWorld.first, fullWorld.second)), B3D(loadMatrix(path_A, true, fullWorld.first, fullWorld.second)),
      C3D() {}

void CombBLAS3DMatrixMultiplication::save_result(std::string path) {
  // nothing
}

void CombBLAS3DMatrixMultiplication::gemm(
    std::vector<size_t> serialized_sizes_B_bytes, size_t max_size_B_bytes) {
  C3D = std::move(multiply(A3D, B3D, fullWorld.first, false, true));
}

} // namespace mults
