#include "utils.hpp"

#include <iostream>
#include <string>

// Source code from: https://www.geeksforgeeks.org/euclidean-algorithms-basic-and-extended/
size_t greatest_common_divider_rec(size_t a, size_t b) {
  if (a == 0) {
    return b;
  }
  return greatest_common_divider_rec(b % a, a);
}

namespace utils {

// Computes the greatest common divider of the given list of values
// The list needs to have at least one element
size_t greatest_common_divider(std::vector<size_t> elems) {
  int divider = 1;
  if (elems.size() >= 1) {
    divider = elems[0];
  }

  if (elems.size() > 1) {
    for (int i = 1; i < elems.size(); ++i) {
      divider = greatest_common_divider_rec(divider, elems[i]);
    }
  }
  return divider;
}

#ifndef NDEBUG
void visualize_raw(double *data, midx_t height, midx_t width,
                   const std::string &name) {
  for (midx_t i = 0; i < height; i++) {
    std::cout << name << " " << i << ":\t";
    for (midx_t j = 0; j < width; j++) {
      std::cout << data[(i * width) + j] << "\t";
    }
    std::cout << "\n";
  }
}
#else
void visualize_raw(double *data, midx_t height, midx_t width,
                   const std::string &name) {}
#endif

#ifndef NDEBUG
template <typename T>
void visualize(matrix::CSRMatrix<T> &csr, const std::string &name) {
  auto matrix = std::vector<double>(csr.height * csr.width, 0);

  for (midx_t i = 0; i < csr.height; i++) {
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
template <typename T>
void visualize(matrix::CSRMatrix<T> &, const std::string &) {}
void print_partitions(partition::Partitions &, size_t) {}
void print_serialized_sizes(std::vector<size_t> &, size_t) {}
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
  prependBuf_ = new PrependBuffer(originalBuf_, format("[{}]: ", mpi_rank));
  std::cout.rdbuf(prependBuf_);
}

CoutWithMPIRank::~CoutWithMPIRank() {
  // Restore the original buffer and clean up
  std::cout.rdbuf(originalBuf_);
  delete prependBuf_;
}
} // namespace utils
