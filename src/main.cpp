#include "bitmap.hpp"
#include "communication.hpp"
#include "matrix.hpp"
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

void write_to_file(std::string name, std::string path) {
  std::ofstream f(path);
  f << name << std::endl;
  f.close();
}

int main(int argc, char **argv) {
  if (argc < 7) {
    std::cerr
        << "Did not get enough arguments. Expected <matrix_path> <run_path>"
        << std::endl;
  }

  std::string algo_name = argv[1];
  std::string matrix_name = argv[2];
  std::string A_path = std::format("matrices/{}/A.mtx", matrix_name);
  std::string B_path = std::format("matrices/{}/B.mtx", matrix_name);
  std::string C_sparsity_path =
      std::format("matrices/{}/C_sparsity.mtx", matrix_name);

  std::string run_path = argv[3];
  std::string partitions_path = std::format("{}/partitions.csv", run_path);
  std::string A_shuffle_path = std::format("{}/A_shuffle", run_path);
  std::string B_shuffle_path = std::format("{}/B_shuffle", run_path);

  int n_runs = std::stoi(argv[4]);
  int n_warmup = std::stoi(argv[5]);
  bitmap::n_sections = std::stoi(argv[6]);

  // Init MPI
  int rank, n_nodes;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  std::string C_path = std::format("{}/C_{}.mtx", run_path, rank);
  std::string measurements_path =
      std::format("{}/measurements_{}.csv", run_path, rank);

  // Custom cout that prepends MPI rank
#ifndef NDEBUG
  utils::CoutWithMPIRank custom_cout(rank);
  std::cout << "Started with algo " << algo_name << std::endl;
#endif

  measure_point(measure::global, measure::MeasurementEvent::START);

  if (rank == MPI_ROOT_ID) {
    std::string matrix_name_path = std::format("{}/matrix", run_path);
    write_to_file(matrix_name, matrix_name_path);
    std::string algo_name_path = std::format("{}/algo", run_path);
    write_to_file(algo_name, algo_name_path);
  }

  // Load sparsity
  matrix::CSRMatrix C(C_sparsity_path, false);

  // Perform the shuffling
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

  // Determine which rows/colums to run
  std::vector<size_t> keep_rows(&A_shuffle[partitions[rank].start_row],
                                &A_shuffle[partitions[rank].end_row]);

  std::vector<size_t> keep_cols(&B_shuffle[partitions[rank].start_col],
                                &B_shuffle[partitions[rank].end_col]);

  mults::MatrixMultiplication *mult = NULL;
  if (algo_name == "baseline") {
    mult = new mults::Baseline(rank, n_nodes, partitions, A_path, &keep_rows,
                               B_path, &keep_cols);
  } else if (algo_name == "advanced") {
    mult = new mults::BaselineAdvanced(
        rank, n_nodes, partitions, A_path, &keep_rows, B_path, &keep_cols);
  } else if (algo_name == "full") {
    mult = new mults::FullMatrixMultiplication(
        rank, n_nodes, partitions, A_path, &keep_rows, B_path, &keep_cols);
  } else if (algo_name == "comb") {
    C_path = std::format("{}/C.mtx", run_path);
    mult = new mults::CombBLASMatrixMultiplication(rank, n_nodes, partitions,
                                                   A_path);
  } else {
    std::cerr << "Unknown algorithm type " << algo_name << "\n";
    exit(1);
  }

  // Share serialization sizes
  std::vector<size_t> serialized_sizes_B_bytes(n_nodes);
  size_t B_byte_size = mult->get_B_serialization_size();
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

  // WARUMP
#ifndef NDEBUG
  std::cout << "Running warmup." << std::endl;
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < n_warmup; i++) {
    // TODO: Check if we use warmup runs
    mult->reset();
    mult->gemm(serialized_sizes_B_bytes, max_B_bytes_size);
  }
  measure::Measure::get_instance()->reset_bytes();
#ifndef NDEBUG
  std::cout << "Finished warmup, performing " << n_warmup << " runs."
            << std::endl;
#endif

  // ACTUAL COMPUTATION!!
  for (int i = 0; i < n_runs; i++) {
    mult->reset();
    communication::sync_start_time(rank);
    measure_point(measure::gemm, measure::MeasurementEvent::START);
    mult->gemm(serialized_sizes_B_bytes, max_B_bytes_size);
    measure_point(measure::gemm, measure::MeasurementEvent::END);
  }

  measure_point(measure::global, measure::MeasurementEvent::END);
#ifndef NDEBUG
  std::cout << "Finished computation.\n";
#endif

  // Store the result
  mult->save_result(C_path);
  measure::Measure::get_instance()->save(measurements_path);

  MPI_Finalize();
}
