#include "utils.hpp"

#include <format>
#include <iostream>
#include <string>

namespace utils {

#ifndef NDEBUG
void visualize(matrix::CSRMatrix &csr) {

  auto matrix = (double *)calloc(csr.height * csr.width, sizeof(double));

  for (size_t i = 0; i < csr.height; i++) {
    auto pos = csr.row_ptr[i];
    auto end = csr.row_ptr[i + 1];

    while (pos < end) {
      auto j = csr.col_idx[pos];
      auto val = csr.values[pos];
      matrix[(i * csr.width) + j] = val;
      ++pos;
    }
  }

  for (size_t i = 0; i < csr.height; i++) {
    std::cout << i << ":\t";
    for (size_t j = 0; j < csr.width; j++) {
      std::cout << matrix[(i * csr.width) + j] << "\t";
    }
    std::cout << "\n";
  }
}

void print_partitions(partition::Partitions &parts, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << "Partition for machine " << i;
    std::cout << ". Rows: [" << parts[i].start_row;
    std::cout << ", " << parts[i].end_row;
    std::cout << "). Columns: [" << parts[i].start_col;
    std::cout << ", " << parts[i].end_row;
    std::cout << ").\n";
  }
}
#else
// We hope this get optimized away :)
void visualize(matrix::CSRMatrix &csr) {}
void print_partitions(partition::Partitions &parts, int size) {}
#endif

// Custom stream buffer that prepends a string to each line of output
PrependBuffer::PrependBuffer(std::streambuf *originalBuf,
                             const std::string &prefix)
    : originalBuf_(originalBuf), prefix_(prefix), isNewLine_(true) {}

int PrependBuffer::overflow(int ch) {
  if (isNewLine_) {
    // Output the prefix before every new line
    originalBuf_->sputn(prefix_.c_str(), prefix_.size());
    isNewLine_ = false;
  }

  if (ch == '\n') {
    isNewLine_ = true;
  }

  return originalBuf_->sputc(ch);
}

// Helper class that replaces std::cout
// with a custom buffer that prepends the MPI Rank
CoutWithMPIRank::CoutWithMPIRank(int mpi_rank) {
  originalBuf_ = std::cout.rdbuf();
  prependBuf_ = new PrependBuffer(originalBuf_, std::format("[{}]: ", mpi_rank));
  std::cout.rdbuf(prependBuf_);
}

CoutWithMPIRank::~CoutWithMPIRank() {
  // Restore the original buffer and clean up
  std::cout.rdbuf(originalBuf_);
  delete prependBuf_;
}
} // namespace utils