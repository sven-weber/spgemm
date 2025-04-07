#include "bitmap.hpp"
#include "communication.hpp"
#include "matrix.hpp"
#include "measure.hpp"
#include "mults.hpp"
#include "partition.hpp"
#include "parts.hpp"
#include "utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mpi.h>

void write_to_file(std::string name, std::string path) {
  std::ofstream f(path);
  f << name << std::endl;
  f.close();
}

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cerr
        << "Did not get enough arguments. Expected <matrix_path> <run_path>"
        << std::endl;
    exit(1);
  }

  // Init MPI
  int rank, n_nodes;
  int required_thread_level = MPI_THREAD_FUNNELED;
  int provided_thread_level;
  // Initialize MPI with the required thread support
  if (MPI_Init_thread(&argc, &argv, required_thread_level,
                      &provided_thread_level) != MPI_SUCCESS) {
    printf("Error initializing MPI with threading\n");
    exit(1);
  }
  if (provided_thread_level != required_thread_level) {
    printf("MPI could not provide requested thread level!");
    exit(1);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);

  std::string algo_name = argv[1];
  std::string shuffling_algo = argv[6];
  std::string partitioning_algo = argv[7];
  std::string matrix_name = argv[2];
  std::string matrix_path = argv[10];

  // Whether to load from parallel files
  // (faster for very large matrices)
  std::string parse_str = argv[8];
  bool parallel_loading = false;
  if (parse_str == "true") {
    parallel_loading = true;
    std::cout << "LOADING FROM PARALLEL FILES" << std::endl;
  }

  std::string A_path;
  std::string B_path;
  if (algo_name.starts_with("comb"s) || !parallel_loading) {
    A_path = utils::format("{}/{}/A.mtx", matrix_path, matrix_name);
    B_path = utils::format("{}/{}/A.mtx", matrix_path, matrix_name);
  } else {
    A_path = utils::format("{}/{}/A_{}.mtx", matrix_path, matrix_name, rank);
    B_path = utils::format("{}/{}/A_{}.mtx", matrix_path, matrix_name, rank);
  }

  std::string run_path = argv[3];
  std::string partitions_path = utils::format("{}/partitions.csv", run_path);
  std::string A_shuffle_path = utils::format("{}/A_shuffle", run_path);
  std::string B_shuffle_path = utils::format("{}/B_shuffle", run_path);

  int n_runs = std::stoi(argv[4]);
  int n_warmup = std::stoi(argv[5]);

  // Whether to persist results
  // (takes too long for super large matrices)
  parse_str = argv[9];
  bool persist_results = true;
  if (parse_str == "false") {
    persist_results = false;
    std::cout << "CAUTION: NOT PERSISTING RESULTS" << std::endl;
  }

  std::string C_path = utils::format("{}/C_{}.mtx", run_path, rank);
  std::string measurements_path =
      utils::format("{}/measurements_{}.csv", run_path, rank);

  if (rank == MPI_ROOT_ID) {
    if (const char *omp_num_threads = std::getenv("OMP_NUM_THREADS")) {
      std::cout << "Running with " << omp_num_threads << " threads"
                << std::endl;
    } else {
      std::cout << "Running SINGLE THREADED!" << std::endl;
    }
  }

  // Custom cout that prepends MPI rank
#ifndef NDEBUG
  utils::CoutWithMPIRank custom_cout(rank);
  std::cout << "Started with algo " << algo_name << std::endl;
#endif

  measure_point(measure::global, measure::MeasurementEvent::START);

  if (rank == MPI_ROOT_ID) {
    std::string matrix_name_path = utils::format("{}/matrix", run_path);
    write_to_file(matrix_name, matrix_name_path);
    std::string algo_name_path = utils::format("{}/algo", run_path);
    write_to_file(algo_name, algo_name_path);
  }

  // Load matrix fields for A and B
  matrix::Fields A_fields =
      matrix::utils::read_fields(A_path, false, nullptr, nullptr);
  matrix::Fields B_fields =
      matrix::utils::read_fields(B_path, false, nullptr, nullptr);

  partition::Partitions partitions(n_nodes);
  partition::Shuffle A_shuffle(A_fields.height);
  partition::Shuffle B_shuffle(B_fields.width);
  if (rank == MPI_ROOT_ID && !algo_name.starts_with("comb"s)) {
    std::cout << "STARTING SHUFFLING!" << std::endl;
    if (shuffling_algo == "none") {
      std::iota(A_shuffle.begin(), A_shuffle.end(), 0);
      std::iota(B_shuffle.begin(), B_shuffle.end(), 0);
    } else if (shuffling_algo == "random") {
      A_shuffle = partition::shuffle(A_fields.height);
      B_shuffle = partition::shuffle(B_fields.width);
    } else if (shuffling_algo == "random_rows") {
      A_shuffle = partition::shuffle(A_fields.height);
      std::iota(B_shuffle.begin(), B_shuffle.end(), 0);
    } else {
      std::cerr << "Unknown shuffling algorithm type " << shuffling_algo
                << "\n";
      exit(1);
    }
    partition::save_shuffle(A_shuffle, A_shuffle_path);
    partition::save_shuffle(B_shuffle, B_shuffle_path);
    std::cout << "SHUFFLING FINISHED!" << std::endl << std::flush;

    // Do the partitioning
    std::cout << "Computing \"" << partitioning_algo << "\" partitioning"
              << std::endl;
    measure_point(measure::partition, measure::MeasurementEvent::START);
    if (partitioning_algo == "balanced") {
      partitions = parts::baseline::balanced_partition(A_path, B_fields.width, n_nodes);
    } else if (partitioning_algo == "naive") {
      partitions = parts::baseline::partition(A_fields.height, B_fields.width, n_nodes);
    } else {
      std::cerr << "Unknown partitioning algorithm type " << partitioning_algo
                << "\n";
      exit(1);
    }

    utils::print_partitions(partitions, n_nodes);
    if (persist_results) {
      partition::save_partitions(partitions, partitions_path);
    }
    measure_point(measure::partition, measure::MeasurementEvent::END);
    std::cout << "ROOT NODE FINISHED; NOW EVERYONE WORKS!" << std::endl;
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

  std::cout << "BROADCASTS FINISHED; EVERYONE LOADING MATRICES NOW;"
            << std::endl
            << std::flush;

  mults::MatrixMultiplication *mult = NULL;
  std::vector<size_t> serialized_sizes_B_bytes(n_nodes);
  if (algo_name == "drop_parallel") {
    auto *tmp = new mults::DropParallel(rank, n_nodes, partitions, A_path,
                                        &keep_rows, B_path, &keep_cols);
    mult = tmp;

    // Share bitmaps
    std::vector<std::bitset<N_SECTIONS>> bitmaps(n_nodes);
    MPI_Allgather(&tmp->bitmap, sizeof(std::bitset<N_SECTIONS>), MPI_BYTE,
                  bitmaps.data(), sizeof(std::bitset<N_SECTIONS>), MPI_BYTE,
                  MPI_COMM_WORLD);
    tmp->bitmaps = bitmaps;

    // Share serialization sizes
    std::vector<size_t> B_byte_sizes = tmp->get_B_serialization_sizes();
    MPI_Alltoall(B_byte_sizes.data(), sizeof(size_t), MPI_BYTE,
                 serialized_sizes_B_bytes.data(), sizeof(size_t), MPI_BYTE,
                 MPI_COMM_WORLD);
  } else if (algo_name == "drop_at_once_parallel") {
    auto *tmp = new mults::DropAtOnceParallel(rank, n_nodes, partitions, A_path,
                                              &keep_rows, B_path, &keep_cols);
    mult = tmp;
    // Share bitmaps
    std::vector<std::bitset<N_SECTIONS>> bitmaps(n_nodes);
    MPI_Allgather(&tmp->bitmap, sizeof(std::bitset<N_SECTIONS>), MPI_BYTE,
                  bitmaps.data(), sizeof(std::bitset<N_SECTIONS>), MPI_BYTE,
                  MPI_COMM_WORLD);
    tmp->bitmaps = bitmaps;

    // Share serialization sizes
    std::vector<size_t> B_byte_sizes = tmp->get_B_serialization_sizes();
    MPI_Alltoall(B_byte_sizes.data(), sizeof(size_t), MPI_BYTE,
                 serialized_sizes_B_bytes.data(), sizeof(size_t), MPI_BYTE,
                 MPI_COMM_WORLD);
    tmp->compute_alltoall_data(serialized_sizes_B_bytes);
  } else if (algo_name == "comb1d") {
    C_path = C_path = utils::format("{}/C.mtx", run_path);
    mult = new mults::CombBLASMatrixMultiplication(rank, n_nodes, partitions,
                                                   A_path);
  } else if (algo_name == "comb2d") {
    C_path = C_path = utils::format("{}/C.mtx", run_path);
    mult = new mults::CombBLAS3DMatrixMultiplication(rank, n_nodes, partitions,
                                                   A_path, 1);
  } else if (algo_name == "comb3d") {
    C_path = C_path = utils::format("{}/C.mtx", run_path);
    mult = new mults::CombBLAS3DMatrixMultiplication(rank, n_nodes, partitions,
                                                   A_path, 8);
  } else {
    std::cerr << "Unknown algorithm type " << algo_name << "\n";
    exit(1);
  }

  // Share serialization sizes
  if (algo_name != "drop_parallel" && algo_name != "drop_at_once_parallel") {
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
  std::cout << "STARTING COMPUTATION" << std::endl << std::flush;
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
  if (persist_results) {
    mult->save_result(C_path);
  }
  measure::Measure::get_instance()->save(measurements_path);

  MPI_Finalize();
}