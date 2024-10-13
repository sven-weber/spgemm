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
  int rank, n_nodes;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);

  // Custom cout that prepends MPI rank
  utils::CoutWithMPIRank custom_cout(rank);
#ifndef NDEBUG
  std::cout << "Hello, world." << std::endl;
#endif

  // Load sparsity
  // TODO: Add some print statements with NDEBUG
  matrix::CSRMatrix C(C_path, false);

  if (rank == MPI_ROOT_ID) {
    // Shuffle the rows and columns indices, for better partitioning
    // TODO: Send this around...
    parts::shuffle::set_seed(true);
    int *shuffled_rows = parts::shuffle::shuffle(C.height);
    int *shuffled_cols = parts::shuffle::shuffle(C.width);
  }

  // TODO: Decide which implementation to use

  // Do the partitioning
  partition::Partitions p(n_nodes);
  if (rank == MPI_ROOT_ID) {
    p = parts::baseline::partition(C, n_nodes);
    utils::print_partitions(p, n_nodes);
  }

  // Distribute the partitioning to all machines
  MPI_Bcast(&p[0], sizeof(partition::Partition) * n_nodes, MPI_BYTE,
            MPI_ROOT_ID, MPI_COMM_WORLD);

  // TODO: Load the partial matrices for your rank!!
  if (rank == 0) {
  }
  matrix::CSRMatrix A(A_path, false);

  matrix::CSRMatrix B(B_path, true);

  // Share serialization sizes
  std::vector<size_t> serialized_sizes_B_bytes(n_nodes);

  size_t B_byte_size = B.serialize()->size();
  MPI_Gather(&B_byte_size, sizeof(size_t), MPI_BYTE,
             &serialized_sizes_B_bytes[0], sizeof(size_t), MPI_BYTE,
             MPI_ROOT_ID, MPI_COMM_WORLD);
  MPI_Bcast(&serialized_sizes_B_bytes[0], sizeof(size_t) * n_nodes, MPI_BYTE,
            MPI_ROOT_ID, MPI_COMM_WORLD);

  // Determine maximum size of elements
  size_t max_B_bytes_size = *std::max_element(serialized_sizes_B_bytes.begin(),
                                              serialized_sizes_B_bytes.end());

  if (rank == MPI_ROOT_ID) {
    utils::print_serialized_sizes(serialized_sizes_B_bytes, max_B_bytes_size);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // TODO: Benchmarking

  // Do the multiplication!
  matrix::CSRMatrix partial_C = mults::baseline::spgemm(
      A, B, rank, n_nodes, p, serialized_sizes_B_bytes, max_B_bytes_size);

  // TODO: Revert any shuffling we applied

  // TODO: Materialize the matrix to file with the rank in the name
  MPI_Finalize();
}