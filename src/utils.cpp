#include "utils.hpp"

#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace utils {

#ifndef NDEBUG
void visualize_raw(double *data, size_t height, size_t width,
                   const std::string &name) {
  for (size_t i = 0; i < height; i++) {
    std::cout << name << " " << i << ":\t";
    for (size_t j = 0; j < width; j++) {
      std::cout << data[(i * width) + j] << "\t";
    }
    std::cout << "\n";
  }
}
#else
void visualize_raw(double *data, size_t height, size_t width,
                   const std::string &name) {}
#endif

#ifndef NDEBUG
void visualize(matrix::CSRMatrix &csr, const std::string &name) {
  auto matrix = std::vector<double>(csr.height * csr.width, 0);

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

  visualize_raw(matrix.data(), csr.height, csr.width, name);
}

void print_partitions(partition::Partitions &parts, size_t size) {
  for (size_t i = 0; i < size; i++) {
    std::cout << "Partition for machine " << i << ". Rows: ["
              << parts[i].start_row << ", " << parts[i].end_row
              << "). Columns: [" << parts[i].start_col << ", "
              << parts[i].end_col << ").\n";
  }
}

void print_serialized_sizes(std::vector<size_t> &sizes, size_t max_size) {
  for (size_t i = 0; i < sizes.size(); i++) {
    std::cout << "Serialized size " << i << ": " << sizes[i] << std::endl;
  }
  std::cout << "Max size: " << max_size << std::endl;
}
#else
// We hope this get optimized away :)
void visualize(matrix::CSRMatrix &csr) {}
void print_partitions(partition::Partitions &parts, int size) {}
void print_serialized_sizes(std::vector<size_t> &sizes, size_t max_size) {}
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
  prependBuf_ =
      new PrependBuffer(originalBuf_, std::format("[{}]: ", mpi_rank));
  std::cout.rdbuf(prependBuf_);
}

CoutWithMPIRank::~CoutWithMPIRank() {
  // Restore the original buffer and clean up
  std::cout.rdbuf(originalBuf_);
  delete prependBuf_;
}
} // namespace utils
