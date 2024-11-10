#include "mults.hpp"

namespace mults {

MatrixMultiplication::MatrixMultiplication(int rank, int n_nodes,
                                           partition::Partitions partitions)
    : rank(rank), n_nodes(n_nodes), partitions(partitions) {}

CSRMatrixMultiplication::CSRMatrixMultiplication(
    int rank, int n_nodes, partition::Partitions partitions, std::string path_A,
    std::vector<size_t> *keep_rows, std::string path_B,
    std::vector<size_t> *keep_cols)
    : MatrixMultiplication(rank, n_nodes, partitions),
      part_A(path_A, false, keep_rows), first_part_B(path_B, true, keep_cols),
      cells(part_A.height, partitions[n_nodes - 1].end_col) {
        drop_sections = bitmap::compute_drop_sections(part_A);
      }

void CSRMatrixMultiplication::save_result(std::string path) {
  matrix::CSRMatrix result(cells);
  result.save(path);
}

size_t CSRMatrixMultiplication::get_B_serialization_size() {
  return first_part_B.serialize()->size();
}

void CSRMatrixMultiplication::reset() {
  cells =
      std::move(matrix::Cells(part_A.height, partitions[n_nodes - 1].end_col));
}

} // namespace mults
