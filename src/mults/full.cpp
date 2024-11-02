#include "communication.hpp"
#include "mpi.h"
#include "mults.hpp"

namespace mults {

FullMatrixMultiplication::FullMatrixMultiplication(
    int rank, int n_nodes, partition::Partitions partitions, std::string path_A,
    std::vector<size_t> *keep_rows, std::string path_B,
    std::vector<size_t> *keep_cols)
    : MatrixMultiplication(rank, n_nodes, partitions),
      part_A(path_A, false, keep_rows), first_part_B(path_B, true, keep_cols),
      result(part_A.height, partitions[n_nodes - 1].end_col) {}

void FullMatrixMultiplication::save_result(std::string path) {
  result.save(path);
}

size_t FullMatrixMultiplication::get_B_serialization_size() {
  return first_part_B.serialize()->size();
}

void FullMatrixMultiplication::reset() {
  result =
      std::move(matrix::Matrix(part_A.height, partitions[n_nodes - 1].end_col));
}

void FullMatrixMultiplication::gemm(
    std::vector<size_t> serialized_sizes_B_bytes, size_t max_size_B_bytes) {
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

    // Matrix multiplication
    for (size_t row = 0; row < part_A.height; row++) {
      // Note: B is transposed!
      double *row_data = &part_A.data[row * part_A.width];
      for (size_t col = 0; col < part_B->height; col++) {
        // TODO: check that part_B->width is what we expect
        double *col_data = &part_B->data[col * part_B->width];
        double res = 0;
        size_t col_elem = 0, row_elem = 0;
        for (size_t i = 0; i < part_A.width; ++i)
          res += row_data[i] * col_data[i];

        // TODO: Why are we producing nulls?
        if (res != 0) {
          auto i = row;
          auto j = partitions[current_rank_B].start_col + col;
          auto idx = (i * part_B->width) + j;
          result.data[idx] = res;
        }
      }
    }

    // Wait for the communication to finish
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    // Deserialize Matrix for next round and switch buffer pointers
    std::swap(received_B_buffer, receiving_B_buffer);
    matrix::Matrix received(received_B_buffer);
    part_B = &received;

    // Get next targets
    send_rank = send_rank != n_nodes - 1 ? send_rank + 1 : 0;
    current_rank_B = recv_rank;
    recv_rank = recv_rank != 0 ? recv_rank - 1 : n_nodes - 1;
  }
}

} // namespace mults
