//
// This file is part of BigWham.
//
// Created by Carlo Peruzzo on 10.01.21.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications :: Jan. 12 2021

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "io/bigwham_io_gen.h"

namespace py = pybind11;
using namespace bie;
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
  };
  /* --------------------------------------------------------------------------
   */

  std::vector<double> &getgetValList() { return this->val_list; };
  std::vector<int> &getgetColumnN() { return this->columN; };
  std::vector<int> &getgetRowN() { return this->rowN; };

  // the following lines are
  // taken from https://github.com/pybind/pybind11/issues/1042
  py::array getRowN() {
    auto v = new std::vector<int>(getgetRowN());
    this->rowN = std::vector<int>();
    auto capsule = py::capsule(
        v, [](void *v) { delete reinterpret_cast<std::vector<int> *>(v); });
    return py::array(v->size(), v->data(), capsule);
  };

  py::array getColumnN() {
    auto v = new std::vector<int>(getgetColumnN());
    this->columN = std::vector<int>();
    auto capsule = py::capsule(
        v, [](void *(v)) { delete reinterpret_cast<std::vector<int> *>(v); });
    return py::array(v->size(), v->data(), capsule);
  };

  py::array getValList() {
    auto v = new std::vector<double>(getgetValList());
    this->val_list = std::vector<double>();
    auto capsule = py::capsule(
        v, [](void *v) { delete reinterpret_cast<std::vector<double> *>(v); });
    return py::array(v->size(), v->data(), capsule);
  };
};
/* -------------------------------------------------------------------------- */

template <typename T>
void declare_array(py::module &m, const std::string &typestr) {
  using Class = il::Array2D<T>;
  std::string pyclass_name = typestr;
  py::class_<Class>(m, pyclass_name.c_str())
      .def(py::init<il::int_t, il::int_t, const T &>())
      .def("shape", &Class::size)
      .def("__setitem__", [](Class &instance, std::array<il::int_t, 2> i,
                             const T &value) { instance(i[0], i[1]) = value; })
      .def("__getitem__",
           [](const Class &instance, std::array<il::int_t, 2> i) {
             return instance(i[0], i[1]);
           })
      .def("__repr__", [](const Class &instance) {
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
      });
}
/* -------------------------------------------------------------------------- */

PYBIND11_MODULE(py_bigwham, m) {

  //    // Binding the mother class Bigwhamio
  //    // option py::dynamic_attr() added to allow new members to be created
  //    dynamically);
  py::class_<BigWhamIOGen>(m, "BigWhamIOSelf", py::dynamic_attr(),
                           py::module_local())
      .def(py::init<>()) // constructor
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
            auto capsule = py::capsule(v, [](void *v) {
              delete reinterpret_cast<std::vector<double> *>(v);
            });
            return py::array(v->size(), v->data(), capsule);
          },
          " dot product between hmat and a vector x", py::arg("x"),
          py::return_value_policy::reference)

      //.def("hdotProduct",            &BigWhamIOGen::matvect, " dot product
      // between hmat and a vector x",py::arg("x"))
      // I change the previous binding of matvect to return a numpy array!!
      // todo: is it possible to move the result of the dot product to an
      // std::array? the array is contiguous in memory but not the vector!!!!!!
      // CP
      .def(
          "matvec",
          [](BigWhamIOGen &self,
             const std::vector<double> &x) -> decltype(auto) {
            auto v = new std::vector<double>(self.MatVec(x));
            auto capsule = py::capsule(v, [](void *v) {
              delete reinterpret_cast<std::vector<double> *>(v);
            });
            return py::array(v->size(), v->data(), capsule);
          },
          " dot product between hmat and a vector x", py::arg("x"),
          py::return_value_policy::reference)

      //      .def("computeStresses", &BigWhamIOGen::computeStresses,
      //           "function to compute the stress at a given set of points")
      //      .def("computeDisplacements", &BigWhamIOGen::computeDisplacements)
      .def("get_hmat_time", &BigWhamIOGen::hmat_time);
  //      .def("getBlockClstrTime", &BigWhamIOGen::getBlockClstrTime)
  //      .def("getBinaryClstrTime", &BigWhamIOGen::getBinaryClstrTime);

  /* --------------------------------------------------------------------------
   */
  py::class_<PyGetFullBlocks>(m, "PyGetFullBlocks")
      .def(py::init<>())
      .def("set", &PyGetFullBlocks::set)
      .def("get_val_list", &PyGetFullBlocks::getValList)
      .def("get_col", &PyGetFullBlocks::getColumnN)
      .def("get_row", &PyGetFullBlocks::getRowN);
  /* --------------------------------------------------------------------------
   */

  declare_array<double>(m, "Real2D");
  declare_array<il::int_t>(m, "Int2D");

  // py::class_<bie::Segment<0>>(m, "Segment0").def(py::init<>());

  /* --------------------------------------------------------------------------
   */
  py::class_<BigWhamIOGen>(m, "BigWhamIORect", py::dynamic_attr())
      .def(py::init<>()) 
      .def("hmat_destructor", &BigWhamIOGen::HmatrixDestructor)
      .def("set", &BigWhamIOGen::Set)
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
          "matvec",
          [](BigWhamIOGen &self,
             const std::vector<double> &x) -> decltype(auto) {
            auto v = new std::vector<double>(self.MatVec(x));
            auto capsule = py::capsule(v, [](void *v) {
              delete reinterpret_cast<std::vector<double> *>(v);
            });
            return py::array(v->size(), v->data(), capsule);
          },
          " dot product between hmat and a vector x", py::arg("x"),
          py::return_value_policy::reference)
      .def("get_hmat_time", &BigWhamIOGen::hmat_time);
}
