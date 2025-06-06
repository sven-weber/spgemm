#include "communication.hpp"
#include "mults.hpp"
#include "utils.hpp"
#include <cstring>
#include <measure.hpp>
#include <unistd.h>

namespace mults {

MPI_Datatype init_custom_mpi_type(int data_multiple_of_size) {
  MPI_Datatype continous_bytes;
  MPI_Type_contiguous(data_multiple_of_size, MPI_BYTE, &continous_bytes);
  MPI_Type_commit(&continous_bytes);
  return std::move(continous_bytes);
}

DropAtOnceParallel::DropAtOnceParallel(int rank, int n_nodes,
                                       partition::Partitions partitions,
                                       std::string path_A,
                                       std::vector<midx_t> *keep_rows,
                                       std::string path_B,
                                       std::vector<midx_t> *keep_cols)
    : MatrixMultiplication(rank, n_nodes, partitions),
      part_A(path_A, false, keep_rows), first_part_B(path_B, keep_cols),
      cells(part_A.height, partitions[n_nodes - 1].end_col),
      bitmap(bitmap::compute_bitmap(part_A)),
      // In the CSR Matrix class we ensure everything is 8 bytes aligned!
      data_multiple_of_size(8), all_to_all_type(init_custom_mpi_type(8)) {}

void DropAtOnceParallel::save_result(std::string path) {
  matrix::ManagedCSRMatrix result(cells);
  result.save(path);
}

size_t DropAtOnceParallel::get_B_serialization_size() {
  return std::get<1>(first_part_B.serialize());
}

std::vector<size_t> DropAtOnceParallel::get_B_serialization_sizes() {
  std::vector<size_t> serialization_sizes(n_nodes);

  for (int i = 0; i < n_nodes; i++) {
    auto start = send_buf.size();
    // TODO: can we know the size before hand?
    // we should be able to since we've alreay done this call in
    // get_B_serialization_sizes
    // Note: We need to divide by data_multiple_of_size
    // to make sure the number fits into int_32
    // We send using the `all_to_all_type` instead of byte
    assert(send_buf.size() % data_multiple_of_size == 0);
    send_displs.push_back(send_buf.size() / data_multiple_of_size);

    measure_point(measure::filter, measure::MeasurementEvent::START);
    auto send_blocks = first_part_B.filter(bitmaps[i]);
    measure_point(measure::filter, measure::MeasurementEvent::END);
    if (i != rank) {
      send_buf.insert(send_buf.end(), send_blocks.begin(), send_blocks.end());
    }

    serialization_sizes[i] = send_blocks.size();
    auto end = send_buf.size();
    assert((end - start) % data_multiple_of_size == 0);
    send_counts.push_back((end - start) / data_multiple_of_size);
    // send_counts.push_back(send_blocks.size());
  }

  return serialization_sizes;
}

void DropAtOnceParallel::reset() {
  cells =
      std::move(matrix::Cells(part_A.height, partitions[n_nodes - 1].end_col));
}

void DropAtOnceParallel::compute_alltoall_data(
    std::vector<size_t> serialized_sizes_B_bytes) {

  size_t displ = 0;
  for (int i = 0; i < n_nodes; i++) {
    // Note: We need to divite by data_multiple_of_size
    // to make sure the number fits into int_32
    // We send using the `all_to_all_type` instead of byte
    assert(displ % data_multiple_of_size == 0);
    recv_displs.push_back(displ / data_multiple_of_size);

    size_t b = 0;
    if (i != rank) {
      b = serialized_sizes_B_bytes[i];
      recv_buf.resize(recv_buf.size() + b);
      displ += b;
    }

    assert(b % data_multiple_of_size == 0);
    recv_counts.push_back(b / data_multiple_of_size);
  }
}

void DropAtOnceParallel::gemm(std::vector<size_t> serialized_sizes_B_bytes,
                              size_t max_size_B_bytes) {
  // Communication
  measure_point(measure::wait_all, measure::MeasurementEvent::START);
  communication::alltoallv_continuous(
      data_multiple_of_size, send_buf.data(), send_counts.data(),
      send_displs.data(), all_to_all_type, recv_buf.data(), recv_counts.data(),
      recv_displs.data(), all_to_all_type, MPI_COMM_WORLD);
  measure_point(measure::wait_all, measure::MeasurementEvent::END);

  // Do computation. We need n-1 computation rounds
  for (int i = 0; i < n_nodes; i++) {
    measure_point(measure::deserialize, measure::MeasurementEvent::START);
    matrix::BlockedCSRMatrix<> part_B =
        (i != rank) ? matrix::BlockedCSRMatrix<>(
                          {&recv_buf[recv_displs[i] * data_multiple_of_size],
                           recv_counts[i] * data_multiple_of_size})
                    : first_part_B;
    measure_point(measure::deserialize, measure::MeasurementEvent::END);

    measure_point(measure::mult, measure::MeasurementEvent::START);
    // Matrix multiplication
#pragma omp parallel for
    for (midx_t row = 0; row < part_A.height; row++) {
      auto [row_data_A, row_pos_A, row_len_A] = part_A.row(row);
      for (midx_t row_elem = 0; row_elem < row_len_A; row_elem++) {
        auto [row_data_B, row_pos_B, row_len_B] =
            part_B.row(row_pos_A[row_elem]);
        for (midx_t col_elem = 0; col_elem < row_len_B; col_elem++) {
          double res = row_data_A[row_elem] * row_data_B[col_elem];
          cells.add({row, partitions[i].start_col + row_pos_B[col_elem]}, res);
        }
      }
    }
    measure_point(measure::mult, measure::MeasurementEvent::END);
  }
}
} // namespace mults