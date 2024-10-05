#include "matrix.hpp"
#include "mpi.h"
#include "mults.hpp"
#include "partition.hpp"
#include "parts.hpp"
#include "utils.hpp"
#include <iostream>

int main(int argc, char **argv) {
  // Init MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Load sparsity
  // TODO: Add some sort of shuffling
  // TODO: Add some print statements with NDEBUG
  matrix::CSRMatrix C("matrices/first/C_sparsity.mtx", false);

  // TODO: Decide which implementation to use

  // Do the partitioning
  partition::Partitions p;
  if (rank == MPI_ROOT_ID) {
    p = parts::baseline::partition(C, size);
  }

  // TODO: Distribute the partitioning
  // MPI_Bcast(p, size * 4, MPI_INT, MPI_ROOT_ID, MPI_COMM_WORLD);

  // TODO: Load the partial matrices for your rank!!
  matrix::CSRMatrix A("matrices/first/A.mtx", false);
  matrix::CSRMatrix B("matrices/first/B.mtx", true);

  utils::visualize(A);
  utils::visualize(B);

  // Do the multiplication!
  matrix::CSRMatrix partial_C = mults::baseline::spgemm(A, B, rank, size, p);

  // TODO: Revert any shuffling we applied

  // TODO: Materialize the matrix to file with the rank in the name
}