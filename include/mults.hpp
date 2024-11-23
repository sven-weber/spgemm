#pragma once

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/SpMat.h"
#include "CombBLAS/SpParMat1D.h"
#include "bitmap.hpp"
#include "matrix.hpp"
#include "partition.hpp"
#include <bitset>

namespace mults {

class MatrixMultiplication {
protected:
  int rank;
  int n_nodes;
  partition::Partitions partitions;

public:
  virtual void gemm(std::vector<size_t>, size_t) {}

  virtual size_t get_B_serialization_size() { return 0; }

  virtual void save_result(std::string) {}

  virtual void reset() {}

  MatrixMultiplication(int rank, int n_nodes, partition::Partitions partitions);
};

class CSRMatrixMultiplication : public MatrixMultiplication {
protected:
  matrix::ManagedCSRMatrix<> part_A;
  matrix::ManagedCSRMatrix<> first_part_B;
  matrix::Cells<> cells;
  std::vector<matrix::section> drop_sections;
  CSRMatrixMultiplication(int rank, int n_nodes,
                          partition::Partitions partitions, std::string path_A,
                          std::vector<midx_t> *keep_rows, std::string path_B,
                          std::vector<midx_t> *keep_cols);

public:
  void save_result(std::string path) override;
  size_t get_B_serialization_size() override;
  void reset() override;
};

class FullMatrixMultiplication : public MatrixMultiplication {
protected:
  matrix::Matrix part_A;
  matrix::Matrix first_part_B;
  matrix::Matrix result;

public:
  FullMatrixMultiplication(int rank, int n_nodes,
                           partition::Partitions partitions, std::string path_A,
                           std::vector<midx_t> *keep_rows, std::string path_B,
                           std::vector<midx_t> *keep_cols);

  void save_result(std::string path) override;
  size_t get_B_serialization_size() override;
  void reset() override;
  void gemm(std::vector<size_t> serialized_sizes_B_bytes,
            size_t max_size_B_bytes) override;
};

class Baseline : public CSRMatrixMultiplication {
public:
  Baseline(int rank, int n_nodes, partition::Partitions partitions,
           std::string path_A, std::vector<midx_t> *keep_rows,
           std::string path_B, std::vector<midx_t> *keep_cols);
  void gemm(std::vector<size_t> serialized_sizes_B_bytes,
            size_t max_size_B_bytes) override;
};

class Outer : public MatrixMultiplication {
protected:
  matrix::ManagedCSRMatrix<> part_A;
  matrix::ManagedCSRMatrix<> first_part_B;
  matrix::Cells<> cells;

public:
  Outer(int rank, int n_nodes, partition::Partitions partitions,
        std::string path_A, std::vector<midx_t> *keep_rows, std::string path_B,
        std::vector<midx_t> *keep_cols);
  void gemm(std::vector<size_t> serialized_sizes_B_bytes,
            size_t max_size_B_bytes) override;

  void save_result(std::string path) override;
  size_t get_B_serialization_size() override;
  void reset() override;
};

class Drop : public MatrixMultiplication {
protected:
  matrix::ManagedCSRMatrix<> part_A;
  matrix::BlockedCSRMatrix<> first_part_B;
  matrix::Cells<> cells;

public:
  std::bitset<N_SECTIONS> bitmap;
  std::vector<std::bitset<N_SECTIONS>> bitmaps;

  Drop(int rank, int n_nodes, partition::Partitions partitions,
       std::string path_A, std::vector<midx_t> *keep_rows, std::string path_B,
       std::vector<midx_t> *keep_cols);
  void gemm(std::vector<size_t> serialized_sizes_B_bytes,
            size_t max_size_B_bytes) override;

  void save_result(std::string path) override;
  size_t get_B_serialization_size() override;
  std::vector<size_t> get_B_serialization_sizes();
  void reset() override;
};

typedef combblas::SpParMat1D<int64_t, double,
                             combblas::SpDCCols<int64_t, double>>
    Sp1D;
typedef combblas::SpParMat<int64_t, double, combblas::SpDCCols<int64_t, double>>
    Sp2D;
typedef combblas::PlusTimesSRing<double, double> PTFF;
class CombBLASMatrixMultiplication : public MatrixMultiplication {
protected:
  shared_ptr<combblas::CommGrid> fullWorld;
  Sp1D A1D;
  Sp1D B1D;
  Sp1D C1D;

public:
  CombBLASMatrixMultiplication(int rank, int n_nodes,
                               partition::Partitions partitions,
                               std::string path_A);
  void save_result(std::string path) override;
  void gemm(std::vector<size_t> serialized_sizes_B_bytes,
            size_t max_size_B_bytes) override;
};

} // namespace mults
