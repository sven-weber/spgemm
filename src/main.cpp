#include "matrix.hpp"
#include "mpi.h"
#include "mults.hpp"
#include "partition.hpp"
#include "parts.hpp"
#include "utils.hpp"
#include <format>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Did not get enough arguments. Expected <matrix_path>";
  }

  std::string matrix_path = argv[1];
  std::string A_path = std::format("{}/A.mtx", matrix_path);
  std::string B_path = std::format("{}/B.mtx", matrix_path);
  std::string C_path = std::format("{}/C_sparsity.mtx", matrix_path);

  // Init MPI
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Custom cout that prepends MPI rank
  utils::CoutWithMPIRank custom_cout(rank);
#ifndef NDEBUG
  std::cout << "Hello, world." << std::endl;
#endif

  // Load sparsity
  // TODO: Add some print statements with NDEBUG
  matrix::CSRMatrix C(C_path, false);
  // Shuffle the rows and columns indices, for better partitioning
  int *shuffled_rows = parts::shuffle::shuffle(C.height);
  int *shuffled_cols = parts::shuffle::shuffle(C.width);

  // TODO: Decide which implementation to use

  // Do the partitioning
  partition::Partitions p(size);
  if (rank == MPI_ROOT_ID) {
    p = parts::baseline::partition(C, size);
    utils::print_partitions(p, size);
  }

  // Distribute the partitioning to all machines
  MPI_Bcast(&p[0], sizeof(partition::Partition) * size, MPI_BYTE, MPI_ROOT_ID,
            MPI_COMM_WORLD);

  // TODO: Load the partial matrices for your rank!!
  matrix::CSRMatrix A(A_path, false);
  matrix::CSRMatrix B(B_path, true);

  std::cout << "A:" << std::endl;
  utils::visualize(A);
  std::cout << "B:" << std::endl;
  utils::visualize(B);

  // Do the multiplication!
  // TODO: Benchmarking
  matrix::CSRMatrix partial_C = mults::baseline::spgemm(A, B, rank, size, p);

  // TODO: Revert any shuffling we applied

  // TODO: Materialize the matrix to file with the rank in the name
  MPI_Finalize();
}