#include "communication.hpp"
#include "mpi.h"
#include "mults.hpp"
#include "utils.hpp"
#include <cstring>
#include <measure.hpp>
#include <unistd.h>

namespace mults {

DropAtOnceParallel::DropAtOnceParallel(int rank, int n_nodes,
                                       partition::Partitions partitions,
                                       std::string path_A,
                                       std::vector<midx_t> *keep_rows,
                                       std::string path_B,
                                       std::vector<midx_t> *keep_cols)
    : MatrixMultiplication(rank, n_nodes, partitions),
      part_A(path_A, false, keep_rows), first_part_B(path_B, keep_cols),
      cells(part_A.height, partitions[n_nodes - 1].end_col),
      bitmap(bitmap::compute_bitmap(part_A)) {}

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
    send_displs.push_back(send_buf.size());

    measure_point(measure::filter, measure::MeasurementEvent::START);
    auto send_blocks = first_part_B.filter(bitmaps[i]);
    measure_point(measure::filter, measure::MeasurementEvent::END);
    if (i != rank)
      send_buf.insert(send_buf.end(), send_blocks.begin(), send_blocks.end());

    serialization_sizes[i] = send_blocks.size();
    auto end = send_buf.size();
    send_counts.push_back(end - start);
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

  int displ = 0;
  for (int i = 0; i < n_nodes; i++) {
    recv_displs.push_back(displ);

    size_t b = 0;
    if (i != rank) {
      b = serialized_sizes_B_bytes[i];
      recv_buf.resize(recv_buf.size() + b);
      displ += b;
    }

    recv_counts.push_back(b);
  }

  communication::alltoallv(send_buf.data(), send_counts.data(),
                           send_displs.data(), MPI_BYTE, recv_buf.data(),
                           recv_counts.data(), recv_displs.data(), MPI_BYTE,
                           MPI_COMM_WORLD);
}

void DropAtOnceParallel::gemm(std::vector<size_t> serialized_sizes_B_bytes,
                              size_t max_size_B_bytes) {
  // Do computation. We need n-1 communication rounds
  for (int i = 0; i < n_nodes; i++) {
    measure_point(measure::deserialize, measure::MeasurementEvent::START);
    matrix::BlockedCSRMatrix<> part_B =
        (i != rank) ? matrix::BlockedCSRMatrix<>(
                          {&recv_buf[recv_displs[i]], recv_counts[i]})
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
