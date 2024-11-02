#pragma once

#include "matrix.hpp"
#include "partition.hpp"

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
        
        MatrixMultiplication(int rank, int n_nodes, partition::Partitions partitions);
};

class CSRMatrixMultiplication : public MatrixMultiplication {
    protected:
        matrix::CSRMatrix part_A;
        matrix::CSRMatrix first_part_B;
        matrix::Cells cells;
        CSRMatrixMultiplication(
            int rank, int n_nodes, partition::Partitions partitions,
            std::string path_A, std::vector<size_t>* keep_rows,
            std::string path_B, std::vector<size_t>* keep_cols); 
    
    public:
        void save_result(std::string path) override;
        size_t get_B_serialization_size() override;
};

class Baseline : public CSRMatrixMultiplication {
public:
    Baseline(int rank, int n_nodes, partition::Partitions partitions,
        std::string path_A, std::vector<size_t>* keep_rows,
        std::string path_B, std::vector<size_t>* keep_cols);
    void gemm(std::vector<size_t> serialized_sizes_B_bytes, size_t max_size_B_bytes);
};

} // namespace mults