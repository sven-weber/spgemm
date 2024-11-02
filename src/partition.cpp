#include "partition.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>

namespace partition {

Shuffle shuffle(size_t size) {
  auto shuffled = std::vector<size_t>(size);

  for (size_t i = 0; i < size; i++) {
    shuffled[i] = i;
  }

#ifndef NSHUFFLE
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffled.begin(), shuffled.end(), g);
#endif

#ifndef NDEBUG
  std::cout << "Shuffled indices: ";
  for (size_t i = 0; i < size; i++) {
    std::cout << shuffled[i] << " ";
  }
  std::cout << std::endl;
#endif

  return shuffled;
}

void save_partitions(Partitions &partitions, std::string file) {
  auto csv = std::ofstream(file);
  assert(!csv.fail());

  csv << "rank,start_row,end_row,start_col,end_col" << std::endl;

  for (size_t i = 0; i < partitions.size(); ++i) {
    auto part = partitions[i];
    csv << i << "," << part.start_row << "," << part.end_row << ","
        << part.start_col << "," << part.end_col << std::endl;
  }

  csv.close();
}

void save_shuffle(Shuffle &shuffle, std::string file) {
  auto map = std::ofstream(file);
  assert(!map.fail());

  for (auto i : shuffle)
    map << i << std::endl;

  map.close();
}

} // namespace partition
