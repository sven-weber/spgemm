#pragma once

#include <mpi.h>

namespace communication {

int send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
         MPI_Comm comm, MPI_Request *request);

int recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
         MPI_Comm comm, MPI_Request *request);

int alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
              MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
              const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm);

int alltoallv_continuous(int number_of_bytes_per_count, const void *sendbuf, const int sendcounts[], const int sdispls[],
              MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
              const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm);

void sync_start_time(int rank);

} // namespace communication
