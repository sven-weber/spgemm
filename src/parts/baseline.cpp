#include "parts.hpp"

namespace parts {
namespace baseline {
partition::Partitions partition(matrix::Matrix &C, int mpi_size) {
    partition::Partitions p (mpi_size);

    // Baseline partition that follows 1D partitioning
    for (int i = 0; i < mpi_size; i++) {
        p[i].start_row = i * C.width / mpi_size;
        p[i].end_row = (i + 1) * C.width / mpi_size;
        p[i].start_col = i * C.height / mpi_size;
        p[i].end_col = (i + 1) * C.height / mpi_size;
    }
}
} // namespace baseline
} // namespace parts