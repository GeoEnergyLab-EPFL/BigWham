//
// This file is part of BigWham.
//
// Created by Carlo Peruzzo on 10.01.21.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications :: Jan. 12 2021

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "io/bigwham_io_gen.h"

namespace nb = nanobind;
template <typename T>
using pyarray = nb::ndarray<nb::numpy, T, nb::shape<nb::any>>;
/* -------------------------------------------------------------------------- */

// get fullBlocks
class PyGetFullBlocks {
private:
  std::vector<double> val_list;
  std::vector<int> rowN;
  std::vector<int> columN;

public:
  PyGetFullBlocks() = default;
  ~PyGetFullBlocks() = default;

  void set(const BigWhamIOGen &BigwhamioObj) {
    std::vector<int> pos_list;
    int nbfentry;

    std::cout << " calling getFullBlocks \n";
    BigwhamioObj.GetFullBlocks(this->val_list, pos_list);
    std::cout << " n entries: " << (this->val_list.size()) << "\n";
    std::cout << " Preparing the vectors \n";

    nbfentry = this->val_list.size();
    this->rowN.resize(nbfentry);
    this->columN.resize(nbfentry);

    for (int i = 0; i < nbfentry; i++) {
      this->rowN[i] = pos_list[2 * i];
      this->columN[i] = pos_list[2 * i + 1];
    }
    std::cout << " --- set pyGetFullBlocks completed ---- \n";
  }

  std::vector<double> GetValList() { return this->val_list; };
  std::vector<int> GetColumnN() { return this->columN; };
  std::vector<int> GetRowN() { return this->rowN; };
};
/* -------------------------------------------------------------------------- */

NB_MODULE(py_bigwham, m) {

  //    // Binding the mother class Bigwhamio
  //    // option nb::dynamic_attr() added to allow new members to be created
  //    dynamically);
  nb::class_<BigWhamIOGen>(m, "BigWhamIOSelf", nb::dynamic_attr())
      .def(nb::init<>()) // constructor
      .def("hmat_destructor", &BigWhamIOGen::HmatrixDestructor)
      .def("set", &BigWhamIOGen::SetSelf)
      .def("get_collocation_points", &BigWhamIOGen::GetCollocationPoints)
      .def("get_permutation", &BigWhamIOGen::GetPermutation)
      .def("get_compression_ratio", &BigWhamIOGen::GetCompressionRatio)
      .def("get_kernel_name", &BigWhamIOGen::kernel_name)
      .def("get_spatial_dimension", &BigWhamIOGen::spatial_dimension)
      .def("matrix_size", &BigWhamIOGen::MatrixSize)
      .def("get_hpattern", &BigWhamIOGen::GetHPattern)
      .def("convert_to_global", &BigWhamIOGen::ConvertToGlobal)
      .def("convert_to_local", &BigWhamIOGen::ConvertToLocal)
      .def(
          "matvec_permute",
          [](BigWhamIOGen &self,
             const std::vector<double> &x){
            auto v = self.MatVecPerm(x);
            size_t shape[1] = {v.size()};
            return pyarray<double>(v.data(), 1, shape);
          })
      .def(
          "matvec",
          [](BigWhamIOGen &self,
             const std::vector<double> &x) {
            auto v = self.MatVec(x);
            size_t shape[1] = {v.size()};
            return pyarray<double>(v.data(), 1, shape);
          })
      .def("get_hmat_time", &BigWhamIOGen::hmat_time);
/* -------------------------------------------------------------------------- */

  nb::class_<PyGetFullBlocks>(m, "PyGetFullBlocks")
      .def(nb::init<>())
      .def("set", &PyGetFullBlocks::set)
      .def("get_val_list",
           [](PyGetFullBlocks &self) {
             auto v = self.GetValList();
             size_t shape[1] = {v.size()};
             return pyarray<double>(v.data(), 1, shape);
           })
      .def("get_col",
           [](PyGetFullBlocks &self) {
             auto v = self.GetColumnN();
             size_t shape[1] = {v.size()};
             return pyarray<int>(v.data(), 1, shape);
           })
      .def("get_row", [](PyGetFullBlocks &self) {
        auto v = self.GetRowN();
        size_t shape[1] = {v.size()};
        return pyarray<int>(v.data(), 1, shape);
      });
}
