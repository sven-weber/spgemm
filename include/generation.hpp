#pragma once

#include "matrix_impl.hpp"
#include "matrix.hpp"

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpCCols.h"
#include "CombBLAS/SpDCCols.h"
#include "CCGrid.h"
#include "SplitMatDist.h"

#include <cstring>
#include <mpi.h>
#include <algorithm>

namespace generation {

template <typename T>
matrix::triplet_matrix<T> generate(std::string type, unsigned scale, unsigned EDGEFACTOR = 0)
{
  measure_point(measure::mat_generation, measure::MeasurementEvent::START);
  double initiator[4];
  if (type == "ER")
  {
    initiator[0] = .25;
    initiator[1] = .25;
    initiator[2] = .25;
    initiator[3] = .25;
  }
  else if (type == "G500")
  {
    initiator[0] = .57;
    initiator[1] = .19;
    initiator[2] = .19;
    initiator[3] = .05;
    EDGEFACTOR = 16;
  }
  else if (type == "SSCA")
  {
    initiator[0] = .6;
    initiator[1] = .4 / 3;
    initiator[2] = .4 / 3;
    initiator[3] = .4 / 3;
    EDGEFACTOR = 8;
  }
  else
  {
    std::cout << "The initiator parameter - " << type << " - is not recognized." << std::endl;
    exit(1);
  }

  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  combblas::CCGrid CMG(nprocs, 1);
  combblas::SpCCols<midx_t, T> A(*GenMat<midx_t, T>(CMG, scale, EDGEFACTOR, initiator, false));
  measure_point(measure::mat_generation, measure::MeasurementEvent::END);
  measure_point(measure::mat_conversion, measure::MeasurementEvent::START);
  auto arrays = A.GetArrays();
  auto col_pointers = arrays.indarrs[0].addr;
  auto row_indices = arrays.indarrs[1].addr;
  auto values = arrays.numarrs[0].addr;
  
  assert(arrays.indarrs[0].count-1 == A.getncol());
  matrix::triplet_matrix<T> tm;
  tm.nrows = A.getnrow();
  tm.ncols = A.getncol();

  int nnz = A.getnnz();
  tm.rows.resize(nnz);
  tm.cols.resize(nnz);
  tm.vals.resize(nnz);

  int idx = 0;
  for(int i = 0; i < tm.ncols; ++i) {
    int start = col_pointers[i];
    int end = col_pointers[i + 1];
    for(int j = start; j < end; ++j) {
      tm.cols[idx] = i;
      tm.rows[idx] = row_indices[j];
      tm.vals[idx] = values[j];

      tm.nrows = std::max(tm.nrows, tm.rows[idx]);
      idx++;
    }
  }
  measure_point(measure::mat_conversion, measure::MeasurementEvent::START);
  return tm;
}

} // namespace matrix
