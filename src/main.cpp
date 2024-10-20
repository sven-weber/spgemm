#include "matrix.hpp"
#include "mults.hpp"
#include "partition.hpp"
#include "parts.hpp"
#include "utils.hpp"
#include "measure.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <mpi.h>

void write_matrix_name(std::string name, std::string path) {
  std::ofstream f(path);
  f << name << std::endl;
  f.close();
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr
        << "Did not get enough arguments. Expected <matrix_path> <run_path>"
        << std::endl;
  }

  std::string matrix_name = argv[1];
  std::string A_path = std::format("matrices/{}/A.mtx", matrix_name);
  std::string B_path = std::format("matrices/{}/B.mtx", matrix_name);
  std::string C_sparsity_path =
      std::format("matrices/{}/C_sparsity.mtx", matrix_name);

  std::string run_path = argv[2];
  std::string partitions_path = std::format("{}/partitions.csv", run_path);
  std::string A_shuffle_path = std::format("{}/A_shuffle", run_path);
  std::string B_shuffle_path = std::format("{}/B_shuffle", run_path);

  // Init MPI
  int rank, n_nodes;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  std::string C_path = std::format("{}/C_{}.mtx", run_path, rank);
  std::string measurements_path = std::format("{}/measurements_{}.csv", run_path, rank);

  // Custom cout that prepends MPI rank
  utils::CoutWithMPIRank custom_cout(rank);
#ifndef NDEBUG
  std::cout << "Started." << std::endl;
#endif

  measure_point(measure::global, measure::MeasurementEvent::START);

  if (rank == MPI_ROOT_ID) {
    std::string matrix_name_path = std::format("{}/matrix", run_path);
    write_matrix_name(matrix_name, matrix_name_path);
  }

  // Load sparsity
  matrix::CSRMatrix C(C_sparsity_path, false);

  partition::Shuffle A_shuffle(C.height);
  partition::Shuffle B_shuffle(C.width);
  if (rank == MPI_ROOT_ID) {
    A_shuffle = std::move(partition::shuffle(C.height));
    B_shuffle = std::move(partition::shuffle(C.width));

    partition::save_shuffle(A_shuffle, A_shuffle_path);
    partition::save_shuffle(B_shuffle, B_shuffle_path);
  }

  // Broadcast the shuffled rows and columns
  MPI_Bcast(A_shuffle.data(), sizeof(size_t) * A_shuffle.size(), MPI_BYTE,
            MPI_ROOT_ID, MPI_COMM_WORLD);
  MPI_Bcast(B_shuffle.data(), sizeof(size_t) * B_shuffle.size(), MPI_BYTE,
            MPI_ROOT_ID, MPI_COMM_WORLD);

  // TODO: Decide which implementation to use

  // Do the partitioning
  partition::Partitions partitions(n_nodes);
  if (rank == MPI_ROOT_ID) {
    partitions = parts::baseline::partition(C, n_nodes);
    utils::print_partitions(partitions, n_nodes);

    partition::save_partitions(partitions, partitions_path);
  }

  // Distribute the partitioning to all machines
  MPI_Bcast(partitions.data(), sizeof(partition::Partition) * partitions.size(),
            MPI_BYTE, MPI_ROOT_ID, MPI_COMM_WORLD);

  // Load the partial matrices for the rank of this process
  std::vector<size_t> keep_rows(&A_shuffle[partitions[rank].start_row],
                                &A_shuffle[partitions[rank].end_row]);
  matrix::CSRMatrix A(A_path, false, &keep_rows);

  std::vector<size_t> keep_cols(&B_shuffle[partitions[rank].start_col],
                                &B_shuffle[partitions[rank].end_col]);
  matrix::CSRMatrix B(B_path, true, &keep_cols);

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

  measure_point(measure::gemm, measure::MeasurementEvent::START);
  // Do the multiplication!
  auto partial_C =
      mults::baseline::spgemm(A, B, rank, n_nodes, partitions,
                              serialized_sizes_B_bytes, max_B_bytes_size);

  measure_point(measure::gemm, measure::MeasurementEvent::END);
  measure_point(measure::global, measure::MeasurementEvent::END);
#ifndef NDEBUG
  std::cout << "Finished computation.\n";
#endif

  // Store the result
  partial_C.save(C_path);
  measure::Measure::get_instance()->save(measurements_path);

  MPI_Finalize();
}
