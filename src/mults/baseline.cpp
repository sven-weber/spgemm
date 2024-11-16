#include "communication.hpp"
#include "mpi.h"
#include "mults.hpp"
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <measure.hpp>
#include <unistd.h>

namespace mults {

Baseline::Baseline(int rank, int n_nodes, partition::Partitions partitions,
                   std::string path_A, std::vector<midx_t> *keep_rows,
                   std::string path_B, std::vector<midx_t> *keep_cols)
    : CSRMatrixMultiplication(rank, n_nodes, partitions, path_A, keep_rows,
                              path_B, keep_cols) {}

void Baseline::gemm(std::vector<size_t> serialized_sizes_B_bytes,
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
    if (i < n_nodes - 1) {
      // Resize buffer to the correct size (should not free/alloc memory)
      receiving_B_buffer->resize(serialized_sizes_B_bytes[recv_rank]);
      communication::send(serialized->data(), serialized_sizes_B_bytes[rank],
                          MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &requests[0]);
      communication::recv(receiving_B_buffer->data(),
                          serialized_sizes_B_bytes[recv_rank], MPI_BYTE,
                          recv_rank, 0, MPI_COMM_WORLD, &requests[1]);
    }

    measure_point(measure::mult, measure::MeasurementEvent::START);
    // Matrix multiplication
    for (midx_t row = 0; row < part_A.height; row++) {
      // Note: B is transposed!
      auto [row_data, row_pos, row_len] = part_A.row(row);
      for (midx_t col = 0; col < part_B->height; col++) {
        auto [col_data, col_pos, col_len] = part_B->col(col);
        double res = 0;
        midx_t col_elem = 0, row_elem = 0;
        // inner loop for multiplication
        while (col_elem < col_len && row_elem < row_len) {
          if (col_pos[col_elem] < row_pos[row_elem]) {
            col_elem++;
          } else if (col_pos[col_elem] > row_pos[row_elem]) {
            row_elem++;
          } else {
            res += row_data[row_elem] * col_data[col_elem];
            row_elem++;
            col_elem++;
          }
        }
        if (res != 0) {
          cells.add({row, partitions[current_rank_B].start_col + col}, res);
        }
      }
    }
    measure_point(measure::mult, measure::MeasurementEvent::END);

    if (i < n_nodes - 1) {
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
}
} // namespace mults
