#include "measure.hpp"

#include <cassert>
#include <mpi.h>

namespace communication {

int send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
         MPI_Comm comm, MPI_Request *request) {
  assert(datatype == MPI_BYTE);
  measure::Measure::get_instance()->track_bytes(count);
  return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
         MPI_Comm comm, MPI_Request *request) {
  assert(datatype == MPI_BYTE);
  return MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

} // namespace communication