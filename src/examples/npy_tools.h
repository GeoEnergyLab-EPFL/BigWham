#include <il/Array.h>
#include <il/Array2D.h>
#include "cnpy.h"

#ifndef NPY_TOOLS_H_
#define NPY_TOOLS_H_

template <typename T>
std::string print_array2D(const il::Array2D<T> &instance) {
  std::string t1 = "Size : " + std::to_string(instance.size(0)) + " X " +
                   std::to_string(instance.size(1)) + "  \n\n";
  std::string t2 = "";
  for (il::int_t i = 0; i < instance.size(0); ++i) {
    for (il::int_t j = 0; j < instance.size(1); ++j) {
      t2 += std::to_string(instance(i, j)) + " ";
    }
    t2 += "\n";
  }
  return t1 + t2;
}

template <typename T>
il::Array2D<T> copy_array2D(const cnpy::NpyArray &B) {
  il::Array2D<T> A(B.shape[0], B.shape[1]);
  for (uint i = 0; i < B.num_vals; i++) {
    A.Data()[i] = B.data<T>()[i];
  }
  return A;
}

template <typename T> std::string print_array1D(const il::Array<T> &instance) {
  std::string t1 = "Size : " + std::to_string(instance.size()) + "\n\n";
  std::string t2 = "";
  for (il::int_t i = 0; i < instance.size(); ++i) {
    t2 += std::to_string(instance[i]) + " ";
    t2 += "\n";
  }
  return t1 + t2;
}


#endif // NPY_TOOLS_H_
