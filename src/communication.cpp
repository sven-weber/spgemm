#include "measure.hpp"

#include "utils.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
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

unsigned long serialize_time_to_unsigned_long(
    const std::chrono::system_clock::time_point &time_point) {
  // Convert time_point to milliseconds since the Unix epoch and cast to
  // unsigned long
  return static_cast<unsigned long>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          time_point.time_since_epoch())
          .count());
}

std::chrono::system_clock::time_point
deserialize_time_from_unsigned_long(unsigned long time_as_ulong) {
  // Convert the unsigned long (milliseconds since epoch) back to a time_point
  auto duration = std::chrono::milliseconds(time_as_ulong);
  return std::chrono::system_clock::time_point(duration);
}

void sync_start_time(int rank) {
  MPI_Barrier(MPI_COMM_WORLD);

  unsigned long time_to_wait_for;
  if (rank == 0) {
    // Get the current time
    auto current_time = std::chrono::system_clock::now();
    // Calculate the time point 2 seconds in the future
    auto start_time = current_time + std::chrono::seconds(2);
    time_to_wait_for = serialize_time_to_unsigned_long(start_time);
  }

  MPI_Bcast(&time_to_wait_for, sizeof(time_to_wait_for), MPI_BYTE, MPI_ROOT_ID,
            MPI_COMM_WORLD);
  auto clock_time = deserialize_time_from_unsigned_long(time_to_wait_for);

#ifndef NDEBUG
  std::cout << "Waiting until " << clock_time << " to start next run\n";
#endif

  // Busy wait until the target time is reached
  while (std::chrono::system_clock::now() < clock_time) {
    // Busy waiting (doing nothing in the loop)
  }
}

} // namespace communication