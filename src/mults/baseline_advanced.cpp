#include "communication.hpp"
#include "mpi.h"
#include "mults.hpp"
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <measure.hpp>
#include <unistd.h>

namespace mults {

BaselineAdvanced::BaselineAdvanced(int rank, int n_nodes,
                                   partition::Partitions partitions,
                                   std::string path_A,
                                   std::vector<midx_t> *keep_rows,
                                   std::string path_B,
                                   std::vector<midx_t> *keep_cols)
    : MatrixMultiplication(rank, n_nodes, partitions),
      part_A(path_A, false, keep_rows),
      first_part_B(path_B, false, nullptr, keep_cols),
      cells(part_A.height, partitions[n_nodes - 1].end_col) {
  drop_sections = bitmap::compute_drop_sections(part_A);
}

void BaselineAdvanced::save_result(std::string path) {
  matrix::CSRMatrix result(cells);
  result.save(path);
}

size_t BaselineAdvanced::get_B_serialization_size() {
  return first_part_B.serialize()->size();
}

void BaselineAdvanced::reset() {
  cells =
      std::move(matrix::Cells(part_A.height, partitions[n_nodes - 1].end_col));
}
void BaselineAdvanced::gemm(std::vector<size_t> serialized_sizes_B_bytes,
                            size_t max_size_B_bytes) {

  // Buffer where the receiving partitions will be stored
  auto receiving_B_buffer =
      std::make_shared<std::vector<char>>(max_size_B_bytes);
  auto received_B_buffer =
      std::make_shared<std::vector<char>>(max_size_B_bytes);

  // Zero-copy serialized representation of B to send around
  auto serialized = first_part_B.serialize();
  auto part_B = &first_part_B;

  int send_rank = rank != n_nodes - 1 ? rank + 1 : 0;
  int current_rank_B = rank;
  int recv_rank = rank != 0 ? rank - 1 : n_nodes - 1;

  MPI_Request requests[2];
  // Do computation. We need n-1 communication rounds
  for (int i = 0; i < n_nodes; i++) {
    // Resize buffer to the correct size (should not free/alloc memory)
    receiving_B_buffer->resize(serialized_sizes_B_bytes[recv_rank]);
    communication::send(serialized->data(), serialized_sizes_B_bytes[rank],
                        MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &requests[0]);
    communication::recv(receiving_B_buffer->data(),
                        serialized_sizes_B_bytes[recv_rank], MPI_BYTE,
                        recv_rank, 0, MPI_COMM_WORLD, &requests[1]);

    measure_point(measure::mult, measure::MeasurementEvent::START);
    // Matrix multiplication
    for (midx_t row = 0; row < part_A.height; row++) {
      auto [row_data_A, row_pos_A, row_len_A] = part_A.row(row);
      for (midx_t row_elem = 0; row_elem < row_len_A; row_elem++) {
        auto [row_data_B, row_pos_B, row_len_B] =
            part_B->row(row_pos_A[row_elem]);
        for (midx_t col_elem = 0; col_elem < row_len_B; col_elem++) {
          double res = row_data_A[row_elem] * row_data_B[col_elem];
          cells.add(
              {row, partitions[current_rank_B].start_col + row_pos_B[col_elem]},
              res);
        }
      }
    }
    measure_point(measure::mult, measure::MeasurementEvent::END);

    measure_point(measure::wait_all, measure::MeasurementEvent::START);
    // Wait for the communication to finish
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    measure_point(measure::wait_all, measure::MeasurementEvent::END);

    // Deserialize Matrix for next round and switch buffer pointers
    std::swap(received_B_buffer, receiving_B_buffer);
    matrix::CSRMatrix received(received_B_buffer);
    part_B = &received;

    // Get next targets
    send_rank = send_rank != n_nodes - 1 ? send_rank + 1 : 0;
    current_rank_B = recv_rank;
    recv_rank = recv_rank != 0 ? recv_rank - 1 : n_nodes - 1;
  }
}
} // namespace mults
