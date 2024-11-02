#pragma once

#include "matrix.hpp"
#include "partition.hpp"
#include <streambuf>

#define MPI_ROOT_ID 0

namespace utils {

/**
 * Visualizes a raw matrix data
 */
void visualize_raw(double *data, size_t height, size_t width,
                   const std::string &name);

/**
 * Visualizes the matrix for debugging
 */
void visualize(matrix::CSRMatrix &csr, const std::string &name);

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
