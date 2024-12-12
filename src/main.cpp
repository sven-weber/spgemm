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
  if (argc < 8) {
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
  if (MPI_Init_thread(&argc, &argv, required_thread_level, &provided_thread_level) != MPI_SUCCESS) {
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
  std::string matrix_name = argv[2];
  std::string matrix_path = argv[7];
  std::string A_path = utils::format("{}/{}/A_{}.mtx", matrix_path, matrix_name, rank % 256);
  std::string B_path = utils::format("{}/{}/A_{}.mtx", matrix_path, matrix_name, rank % 256);
  std::string C_sparsity_path =
      utils::format("{}/{}/C_sparsity.mtx", matrix_path, matrix_name);
  std::string A_shuffle_path =
      utils::format("{}/{}/A_shuffle", matrix_path, matrix_name);
  std::string B_shuffle_path =
      utils::format("{}/{}/B_shuffle", matrix_path, matrix_name);

  std::string run_path = argv[3];
  std::string partitions_path = utils::format("{}/partitions.csv", run_path);
  std::string A_shuffle_link_path = utils::format("{}/A_shuffle", run_path);
  std::string B_shuffle_link_path = utils::format("{}/B_shuffle", run_path);

  int n_runs = std::stoi(argv[4]);
  int n_warmup = std::stoi(argv[5]);

  // Whether to persist results
  // (takes too long for super large matrices)
  std::string parse_str = argv[6];
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

  // Load sparsity
  matrix::Fields A_fields =
      matrix::utils::read_fields(A_path, false, nullptr, nullptr);
  matrix::Fields B_fields =
      matrix::utils::read_fields(B_path, false, nullptr, nullptr);

  partition::Partitions partitions(n_nodes);
  partition::Shuffle A_shuffle(A_fields.height);
  partition::Shuffle B_shuffle(B_fields.width);
  if (rank == MPI_ROOT_ID && algo_name != "comb") {
    std::cout << "STARTING SHUFFLING!" << std::endl;
    // Shuffling can be expensive (mostly because C needs to be loaded!)
    // Therefore, we persist it!
    bool loaded_A = partition::load_shuffle(A_shuffle_path, A_shuffle);
    bool loaded_B = partition::load_shuffle(B_shuffle_path, B_shuffle);

    if (!loaded_A || !loaded_B) {
      std::cout << "Computing shuffling since no existing one could be found"
                << std::endl;
      // Perform the shuffling if no persistet one exists!
      partition::iterative_shuffle(C_sparsity_path, &A_shuffle, &B_shuffle);
      partition::save_shuffle(A_shuffle, A_shuffle_path);
      partition::save_shuffle(B_shuffle, B_shuffle_path);
    }

    if (persist_results) {
      try {
        std::filesystem::create_symlink("../../" + A_shuffle_path,
                                        A_shuffle_link_path);
        std::filesystem::create_symlink("../../" + B_shuffle_path,
                                        B_shuffle_link_path);
        std::cout << "Symbolic links created successfully." << std::endl;
      } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
      }
    }
    std::cout << "SHUFFLING FINISHED!" << std::endl << std::flush;

    // Do the partitioning

    // Better (but expensive) partitioning algorithm
    // Because the better partitioning requires C (which takes a very long time
    // to load!), we will not use it matrix::ManagedCSRMatrix<short>
    // mat(C_sparsity_path, false, &A_shuffle,
    //                                &B_shuffle);
    // partitions = parts::baseline::balanced_partition(mat, n_nodes);

    measure_point(measure::partition, measure::MeasurementEvent::START);
    auto fields =
        matrix::utils::read_fields(C_sparsity_path, false, nullptr, nullptr);
    partitions = parts::baseline::partition(fields, n_nodes);
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

  std::cout << "BROADCASTS FINISHED; EVERYONE LOADING MATRICES NOW;" << std::endl << std::flush;

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
                  bitmaps.data(), sizeof(std::bitset<N_SECTIONS>), MPI_BYTE,
                  MPI_COMM_WORLD);
    tmp->bitmaps = bitmaps;

    // Share serialization sizes
    std::vector<size_t> B_byte_sizes = tmp->get_B_serialization_sizes();
    MPI_Alltoall(B_byte_sizes.data(), sizeof(size_t), MPI_BYTE,
                 serialized_sizes_B_bytes.data(), sizeof(size_t), MPI_BYTE,
                 MPI_COMM_WORLD);
  } else if (algo_name == "drop_parallel") {
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
  } else if (algo_name == "drop_at_once") {
    auto *tmp = new mults::DropAtOnce(rank, n_nodes, partitions, A_path,
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
  } else if (algo_name == "full") {
    mult = new mults::FullMatrixMultiplication(
        rank, n_nodes, partitions, A_path, &keep_rows, B_path, &keep_cols);
  } else if (algo_name == "comb") {
    C_path = C_path = utils::format("{}/C.mtx", run_path);
    mult = new mults::CombBLASMatrixMultiplication(rank, n_nodes, partitions,
                                                   A_path);
  } else {
    std::cerr << "Unknown algorithm type " << algo_name << "\n";
    exit(1);
  }

  // Share serialization sizes
  if (algo_name != "drop" && algo_name != "drop_at_once" &&
      algo_name != "drop_parallel" && algo_name != "drop_at_once_parallel") {
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
  if (persist_results) {
    mult->save_result(C_path);
  }
  measure::Measure::get_instance()->save(measurements_path);

  MPI_Finalize();
}