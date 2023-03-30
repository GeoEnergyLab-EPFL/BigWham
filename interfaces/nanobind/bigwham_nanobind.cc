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

#include "io/bigwham_io_gen.h"

namespace nb = nanobind;
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

  std::vector<double> &getgetValList() { return this->val_list; };
  std::vector<int> &getgetColumnN() { return this->columN; };
  std::vector<int> &getgetRowN() { return this->rowN; };

  // the following lines are
  // taken from https://github.com/pybind/pybind11/issues/1042
  // nb::ndarray getRowN() {
  //   auto v = new std::vector<int>(getgetRowN());
  //   this->rowN = std::vector<int>();
  //   auto capsule = nb::capsule(
  //       v, [](void *v) { delete reinterpret_cast<std::vector<int> *>(v); });
  //   return nb::array(v->size(), v->data(), capsule);
  // };

  // nb::ndarray getRowN() {
  //   // auto v = new std::vector<int>(getgetRowN());
  //   // this->rowN = std::vector<int>();
  //   // auto capsule = nb::capsule(
  //   //     v, [](void *v) { delete reinterpret_cast<std::vector<int> *>(v);
  //   });
  //   // return nb::array(v->size(), v->data(), capsule);
  //   size_t shape = {v.size()};
  //   return nb::ndarray<nb::numpy, int>(getgetValList().data(), 1, shape);
  // };

  // nb::array getColumnN() {
  //   auto v = new std::vector<int>(getgetColumnN());
  //   this->columN = std::vector<int>();
  //   auto capsule = nb::capsule(
  //       v, [](void *(v)) { delete reinterpret_cast<std::vector<int> *>(v);
  //       });
  //   return nb::array(v->size(), v->data(), capsule);
  // };

  // nb::array getValList() {
  //   auto v = new std::vector<double>(getgetValList());
  //   this->val_list = std::vector<double>();
  //   auto capsule = nb::capsule(
  //       v, [](void *v) { delete reinterpret_cast<std::vector<double> *>(v);
  //       });
  //   return nb::array(v->size(), v->data(), capsule);
  // };
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
      //.def("hdotProductInPermutted", &BigWhamIOGen::hdotProductInPermutted)
      // I change the previous binding of hdotProductInPermutted to return a
      // numpy array!!
      .def(
          "matvec_permute",
          [](BigWhamIOGen &self,
             const std::vector<double> &x) -> decltype(auto) {
            auto v = new std::vector<double>(self.MatVecPerm(x));
            auto capsule = nb::capsule(v, [](void *v) {
              delete reinterpret_cast<std::vector<double> *>(v);
            });
            return nb::array(v->size(), v->data(), capsule);
          },
          " dot product between hmat and a vector x", nb::arg("x"),
          nb::return_value_policy::reference)

      //.def("hdotProduct",            &BigWhamIOGen::matvect, " dot product
      // between hmat and a vector x",nb::arg("x"))
      // I change the previous binding of matvect to return a numpy array!!
      // todo: is it possible to move the result of the dot product to an
      // std::array? the array is contiguous in memory but not the vector!!!!!!
      // CP
      .def(
          "matvec",
          [](BigWhamIOGen &self,
             const std::vector<double> &x) -> decltype(auto) {
            auto v = new std::vector<double>(self.MatVec(x));
            auto capsule = nb::capsule(v, [](void *v) {
              delete reinterpret_cast<std::vector<double> *>(v);
            });
            return nb::array(v->size(), v->data(), capsule);
          },
          " dot product between hmat and a vector x", nb::arg("x"),
          nb::return_value_policy::reference)

      //      .def("computeStresses", &BigWhamIOGen::computeStresses,
      //           "function to compute the stress at a given set of points")
      //      .def("computeDisplacements", &BigWhamIOGen::computeDisplacements)
      .def("get_hmat_time", &BigWhamIOGen::hmat_time);
  //      .def("getBlockClstrTime", &BigWhamIOGen::getBlockClstrTime)
  //      .def("getBinaryClstrTime", &BigWhamIOGen::getBinaryClstrTime);

  nb::class_<PyGetFullBlocks>(m, "PyGetFullBlocks")
      .def(nb::init<>())
      .def("set", &PyGetFullBlocks::set)
      .def("get_val_list",
           [](PyGetFullBlocks &self) {
             auto v = self.getgetValList();
             size_t shape = {v.size()};
             return nb::ndarray<nb::numpy, double>(v.data(), 1, shape);
           })
      .def("get_col", &PyGetFullBlocks::getColumnN)
      .def("get_row", &PyGetFullBlocks::getRowN);
}
