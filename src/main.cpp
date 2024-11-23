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

  // Load sparsity
  matrix::CSRMatrix C(C_sparsity_path, false);

  
  partition::Partitions partitions(n_nodes);
  partition::Shuffle A_shuffle(C.height);
  partition::Shuffle B_shuffle(C.width);
  if (rank == MPI_ROOT_ID) {
    matrix::CSRMatrix A(A_path, false);

    // Perform the shuffling
    A_shuffle = std::move(partition::shuffle_min(A));
    B_shuffle = std::move(partition::shuffle(C.width));

    partition::save_shuffle(A_shuffle, A_shuffle_path);
    partition::save_shuffle(B_shuffle, B_shuffle_path);

    // Do the partitioning
    partitions = parts::baseline::balanced_partition(C, n_nodes);
    utils::print_partitions(partitions, n_nodes);

    partition::save_partitions(partitions, partitions_path);
  }

  // Broadcast the shuffled rows and columns
  MPI_Bcast(A_shuffle.data(), sizeof(midx_t) * A_shuffle.size(), MPI_BYTE,
            MPI_ROOT_ID, MPI_COMM_WORLD);
  MPI_Bcast(B_shuffle.data(), sizeof(midx_t) * B_shuffle.size(), MPI_BYTE,
            MPI_ROOT_ID, MPI_COMM_WORLD);

  // Distribute the partitioning to all machines
  MPI_Bcast(partitions.data(), sizeof(partition::Partition) * partitions.size(),
            MPI_BYTE, MPI_ROOT_ID, MPI_COMM_WORLD);

  // Determine which rows/colums to run
  std::vector<midx_t> keep_rows(&A_shuffle[partitions[rank].start_row],
                                &A_shuffle[partitions[rank].end_row]);

  std::vector<midx_t> keep_cols(&B_shuffle[partitions[rank].start_col],
                                &B_shuffle[partitions[rank].end_col]);

  mults::MatrixMultiplication *mult = NULL;
  std::vector<size_t> serialized_sizes_B_bytes(n_nodes);
  if (algo_name == "baseline") {
    mult = new mults::Baseline(rank, n_nodes, partitions, A_path, &keep_rows,
                               B_path, &keep_cols);
  } else if (algo_name == "outer") {
    mult = new mults::Outer(rank, n_nodes, partitions, A_path, &keep_rows,
                            B_path, &keep_cols);
  } else if (algo_name == "drop") {
    auto *tmp = new mults::Drop(rank, n_nodes, partitions, A_path, &keep_rows,
                            B_path, &keep_cols);
    mult = tmp;

    // Share bitmaps
    std::vector<std::bitset<N_SECTIONS>> bitmaps(n_nodes);
    MPI_Allgather(&tmp->bitmap, sizeof(std::bitset<N_SECTIONS>), MPI_BYTE,
              &bitmaps[0], sizeof(std::bitset<N_SECTIONS>), MPI_BYTE, MPI_COMM_WORLD);
    tmp->bitmaps = bitmaps;

    // Share serialization sizes
    std::vector<size_t> B_byte_sizes = tmp->get_B_serialization_sizes();
    MPI_Alltoall(&B_byte_sizes, sizeof(size_t) , MPI_BYTE,
              &serialized_sizes_B_bytes[0], sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);
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
  if (algo_name != "drop") {
    size_t B_byte_size = mult->get_B_serialization_size();
    MPI_Gather(&B_byte_size, sizeof(size_t), MPI_BYTE,
              &serialized_sizes_B_bytes[0], sizeof(size_t), MPI_BYTE,
              MPI_ROOT_ID, MPI_COMM_WORLD);
    MPI_Bcast(&serialized_sizes_B_bytes[0], sizeof(size_t) * n_nodes, MPI_BYTE,
              MPI_ROOT_ID, MPI_COMM_WORLD);
  }

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
    measure::Measure::get_instance()->flush_bytes();
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
