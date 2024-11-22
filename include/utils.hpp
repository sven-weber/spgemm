#pragma once

#include "matrix.hpp"
#include "partition.hpp"
#include <memory>
#include <streambuf>

#define MPI_ROOT_ID 0

namespace utils {

/**
 * Visualizes a raw matrix data
 */
void visualize_raw(double *data, midx_t height, midx_t width,
                   const std::string &name);

/**
 * Visualizes the matrix for debugging
 */
template <typename T = double, class Allocator = std::allocator<std::byte>>
void visualize(const matrix::CSRMatrix<T, Allocator> &csr,
               const std::string &name)
#ifndef NDEBUG
{
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
#else
{
}
#endif

/**
 * Prints the given partition to std out for debugging
 */
void print_partitions(partition::Partitions &part, size_t size);

void print_serialized_sizes(std::vector<size_t> &sizes, size_t max_size);

// Private class for prepending to buffers
class PrependBuffer : public std::streambuf {
public:
  PrependBuffer(std::streambuf *originalBuf, const std::string &prefix);

protected:
  virtual int overflow(int ch) override;

private:
  std::streambuf *originalBuf_;
  std::string prefix_;
  bool isNewLine_;
};

/**
 * Custom cout replacement that prepends the MPI rank of each process
 */
class CoutWithMPIRank {
public:
  CoutWithMPIRank(int mpi_rank);
  ~CoutWithMPIRank();

private:
  std::streambuf *originalBuf_;
  PrependBuffer *prependBuf_;
};

} // namespace utils
