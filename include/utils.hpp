#pragma once

#include "matrix.hpp"
#include "partition.hpp"
#include <streambuf>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define MPI_ROOT_ID 0

namespace utils {

// Helper to handle the variadic arguments and insert them into the string.
template <typename T>
void formatHelper(std::ostringstream& oss, const std::string& fmt, size_t& pos, T&& arg) {
    // Find the next `{}` placeholder
    size_t openBrace = fmt.find('{', pos);
    size_t closeBrace = fmt.find('}', openBrace);

    if (openBrace == std::string::npos || closeBrace == std::string::npos || closeBrace < openBrace) {
        throw std::runtime_error("Mismatched braces in format string.");
    }

    // Append text before the placeholder
    oss << fmt.substr(pos, openBrace - pos);

    // Insert the argument
    oss << std::forward<T>(arg);

    // Update position to the character after the placeholder
    pos = closeBrace + 1;
}

template <typename T, typename... Args>
void formatHelper(std::ostringstream& oss, const std::string& fmt, size_t& pos, T&& arg, Args&&... args) {
    formatHelper(oss, fmt, pos, std::forward<T>(arg));
    formatHelper(oss, fmt, pos, std::forward<Args>(args)...);
}

template <typename... Args>
std::string format(const std::string& fmt, Args&&... args) {
    std::ostringstream oss;
    size_t pos = 0;

    formatHelper(oss, fmt, pos, std::forward<Args>(args)...);

    // Append remaining text after the last placeholder
    oss << fmt.substr(pos);

    return oss.str();
}

/**
 * Visualizes a raw matrix data
 */
void visualize_raw(double *data, midx_t height, midx_t width,
                   const std::string &name);

/**
 * Visualizes the matrix for debugging
 */
template <typename T = double>
void visualize(matrix::CSRMatrix<T> &csr, const std::string &name);

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
