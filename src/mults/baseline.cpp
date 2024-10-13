#include "mpi.h"
#include "mults.hpp"
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <unistd.h>

void print_matrix(double *matrix, int rows, int cols) {
  std::string print = "";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      print += std::to_string(matrix[i * cols + j]) + " ";
    }
    print += "\n";
  }
  std::cout << print;
}

namespace mults {
namespace baseline {
matrix::CSRMatrix spgemm(matrix::CSRMatrix &part_A, matrix::CSRMatrix &part_B,
                         int rank, int size, partition::Partitions partitions,
                         std::vector<size_t>serialized_sizes_B_bytes, size_t max_size_B_bytes) {
  // Define result matrix
  // TODO: Partition with lucas serialization stuff
  double *result = (double *)malloc(sizeof(double) * 2 * 4);
  // Buffer where the receiving partitions will be stored
  auto temp_B_buffer = std::make_shared<std::vector<char>>(max_size_B_bytes);
  // Zero-copy serialized representation of B to send around
  auto serialized = part_B.serialize();

  // Async send and receive B partitions
  int target = rank == 0 ? 1 : 0;
  MPI_Request send, recv;
  MPI_Isend(serialized->data(), max_size_B_bytes, MPI_BYTE, target, 0, MPI_COMM_WORLD, &send);
  MPI_Irecv(temp_B_buffer->data(), max_size_B_bytes, MPI_BYTE, target, 0, MPI_COMM_WORLD, &recv);
  MPI_Waitall(2, (MPI_Request[]) {send, recv}, MPI_STATUSES_IGNORE);

  matrix::CSRMatrix revc_B = matrix::CSRMatrix(temp_B_buffer);
  utils::visualize(revc_B);
  
  return part_A;
  
  /*
  for (int row = 0; row < part_A.height; row++) {
    // Note: B is transposed!
    auto [row_data, row_pos, row_len] = part_A.row(row);
    for (int col = 0; col < part_B.height; col++) {
      auto [col_data, col_pos, col_len] = part_B.col(col);
      double res = 0;
      int col_elem = 0, row_elem = 0;
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
      result[row * 4 + col] = res;
    }
  }

  std::cout << "Multiplication result:\n";
  print_matrix(result, 2, 4);
*/
}
} // namespace baseline
} // namespace mults

/*
int main(int argc, char **argv) {
    // Broadcast partitioning
    MPI_Bcast(partitioning, size * 4, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Print partitioning
    std::cout << "Partitioning for rank " << rank << ": " << partitioning[rank *
4] << " "
        << partitioning[rank * 4 + 1] << " " << partitioning[rank * 4 + 2] << "
" << partitioning[rank * 4 + 3] << std::endl;

    // TODO : Get matrix from call: each rank will receive the portion of the
matrix they work on int rows_per_partition = partitioning[rank * 4 + 1] -
partitioning[rank * 4]; int *matrix_A = (int*) malloc(sizeof(int) *
rows_per_partition * M);

    int cols_per_partition = partitioning[rank * 4 + 3] - partitioning[rank * 4
+ 2]; int size_B = cols_per_partition * N; int *matrix_B = (int*)
malloc(sizeof(int) * size_B); int *received_B = (int*) malloc(sizeof(int) *
size_B); int *temp_B = (int*) malloc(sizeof(int) * size_B);

    int matrix [N * M] = {0};
    // Fill matrix
    int count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix[i * M + j] = count++;
        }
    }
    for (int i = partitioning[rank * 4]; i < partitioning[rank * 4 + 1]; i++) {
        for (int j = 0; j < M; j++) {
            matrix_A[(i - partitioning[rank * 4]) * M + j] = matrix[i * M + j];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = partitioning[rank * 4 + 2]; j < partitioning[rank * 4 + 3];
j++) { matrix_B[i * cols_per_partition + (j - partitioning[rank * 4 + 2])] =
matrix[i * M + j];
        }
    }

    // Print matrix
    std::cout << "Matrix for rank " << rank << ":\n";
    print_matrix(matrix_A, rows_per_partition, M);
    std::cout << "Matrix B for rank " << rank << ":\n";
    print_matrix(matrix_B, N, cols_per_partition);

    // Copy matrix_B to received_B
    memcpy(received_B, matrix_B, size_B * sizeof(int));

    // Define result matrix
    int *result = (int*) malloc(sizeof(int) * rows_per_partition * M);

    // Barrier: wait for everyone to receive the partitioning
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication
    // Start by computing the local result. At the same time, asynchronously
send the cols of matrix_B to the next
    // logical rank and receive the cols of matrix_B from the previous logical
rank.
    // Then compute the result of the multiplication with the received cols. At
the same time, asynchronously send the
    // cols of matrix_B to the next logical rank and receive the cols of
matrix_B from the previous logical rank.
    // Repeat until all ranks have received all the cols of matrix_B.
    int prev = rank != 0 ? rank - 1 : size - 1;
    int curr = rank;
    int next = rank != size - 1 ? rank + 1 : 0;
    for (int l = 0; l < size; l++) {
        MPI_Request send, recv;
        if (next != rank) {
            // Async send and receive cols
            MPI_Isend(matrix_B, size_B, MPI_INT, next, 0, MPI_COMM_WORLD,
&send); MPI_Irecv(temp_B, size_B, MPI_INT, prev, 0, MPI_COMM_WORLD, &recv);

            std::cout << "Rank " << rank << " sending to " << next << " and
receivng from " << prev << std::endl;
        }

        // Perform multiplication
        for (int i = 0; i < rows_per_partition; i++) {
            for (int j = 0; j < cols_per_partition; j++) {
                int res = 0;
                for (int k = 0; k < M; k++) {
                    res += matrix_A[i * M + k] * received_B[k *
cols_per_partition + j];
                }
                result[i * M + curr * cols_per_partition + j] = res;
            }
        }

        if (next != rank) {
            // Wait for send and receive to finish
            MPI_Waitall(2, (MPI_Request[]) {send, recv}, MPI_STATUSES_IGNORE);
        }

        // Prepare for next iteration
        prev = prev != 0 ? prev - 1 : size - 1;
        curr = curr != size - 1 ? curr + 1 : 0;
        next = next != size - 1 ? next + 1 : 0;

        // Swap pointers
        int *temp = received_B;
        received_B = temp_B;
        temp_B = temp;

        if (rank == 0){
            std::cout << "Received B:\n";
            print_matrix(received_B, N, cols_per_partition);
        }
    }

    // Barrier: wait for everyone to finish
    MPI_Barrier(MPI_COMM_WORLD);

    // Print result
    std::cout << "Result for rank " << rank << ":\n";
    print_matrix(result, rows_per_partition, M);

    // Gather results
    // TODO : Gather results from all ranks


    // Free memory
    free(matrix_A);
    free(matrix_B);
    free(received_B);
    free(temp_B);


}
*/