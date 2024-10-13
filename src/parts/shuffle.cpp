#include "parts.hpp"

#include <iostream>

namespace parts {
namespace shuffle {

void set_seed(bool reproducible) {
  if (reproducible) {
    srand(7);
  } else {
    srand(time(NULL));
  }
}

int *shuffle(int size) {
  int *shuffled = new int[size];
  for (int i = 0; i < size; i++) {
    shuffled[i] = i;
  }
  for (int i = 0; i < size; i++) {
    int j = rand() % size;
    int temp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = temp;
  }

#ifndef NDEBUG
  std::cout << "Shuffled indices: ";
  for (int i = 0; i < size; i++) {
    std::cout << shuffled[i] << " ";
  }
  std::cout << std::endl;
#endif

  return shuffled;
}
} // namespace shuffle
} // namespace parts