#include <fstream>
#include <iostream>

#include "matrix.hpp"

namespace matrix {

Matrix::Matrix(size_t start_i, size_t start_j)
    : start_i(start_i), start_j(start_j) {}

CSRMatrix::CSRMatrix(size_t start_i, size_t start_j, std::string file_path)
    : Matrix(start_i, start_j) {
  std::ifstream stream(file_path);

  if (!stream.is_open()) {
    std::cout << "could not open file: " << file_path << std::endl;
    exit(1);
  }

  std::string line;
  do {
    getline(stream, line);
  } while (line.size() > 0 and line[0] == '%');

  // chris scrivi qua

  stream.close();
}

} // namespace matrix
