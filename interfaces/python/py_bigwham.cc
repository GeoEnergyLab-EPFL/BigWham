//
// This file is part of BigWham.
//
// Created by Carlo Peruzzo on 10.01.21.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
// Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
// file for more details.
//
// last modifications :: Jan. 12 2021

#include <memory>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "elements/boundary_element.h"
#include "io/bigwham_io_gen.h"

#include "py_bigwham_helper.h"

namespace py = pybind11;
using namespace bie;
/* -------------------------------------------------------------------------- */

class BigWhamIORect : public BigWhamIOGen {
public:
  BigWhamIORect() {}
  ~BigWhamIORect() {}
};
/* -------------------------------------------------------------------------- */

class PyGetFullBlocks {
private:
  il::Array<double> val_list;
  il::Array<int> rowN;
  il::Array<int> columN;

public:
  PyGetFullBlocks() = default;
  ~PyGetFullBlocks() = default;

  void set(const BigWhamIOGen &BigwhamioObj) {
    il::Array<int> pos_list;
    int nbfentry;

    std::cout << " calling getFullBlocks \n";
    BigwhamioObj.GetFullBlocks(this->val_list, pos_list);
    std::cout << " n entries: " << (this->val_list.size()) << "\n";
    std::cout << " Preparing the vectors \n";

    nbfentry = this->val_list.size();
    this->rowN.Resize(nbfentry);
    this->columN.Resize(nbfentry);

    for (int i = 0; i < nbfentry; i++) {
      this->rowN[i] = pos_list[2 * i];
      this->columN[i] = pos_list[2 * i + 1];
    }
    std::cout << " --- set pyGetFullBlocks completed ---- \n";
  };
  /* --------------------------------------------------------------------------
   */

  pbarray<double> getValList() {
    return as_pyarray<double>(std::move(this->val_list));
  };
  pbarray<int> getColumnN() {
    return as_pyarray<int>(std::move(this->columN));
  };
  pbarray<int> getRowN() { return as_pyarray<int>(std::move(this->rowN)); };
};
/* -------------------------------------------------------------------------- */

PYBIND11_MODULE(py_bigwham, m) {

  //    // Binding the mother class Bigwhamio
  //    // option py::dynamic_attr() added to allow new members to be created
  //    dynamically);
  // Square Self Interaction matrices
  py::class_<BigWhamIOGen>(m, "BigWhamIOSelf", py::dynamic_attr(),
                           py::module_local())
      .def(py::init<>()) // constructor
      .def("hmat_destructor", &BigWhamIOGen::HmatrixDestructor)
      .def("set", py::overload_cast<const std::string &>(&BigWhamIORect::Set))
      .def("set",
           py::overload_cast<const std::vector<double> &,
                             const std::vector<int> &, const std::string &,
                             const std::vector<double> &, const int,
                             const double, const double>(&BigWhamIOGen::Set))
      .def("get_collocation_points", &BigWhamIOGen::GetCollocationPoints)
      .def("get_permutation", &BigWhamIOGen::GetPermutation)
      .def("get_compression_ratio", &BigWhamIOGen::GetCompressionRatio)
      .def("get_kernel_name", &BigWhamIOGen::kernel_name)
      .def("get_spatial_dimension", &BigWhamIOGen::spatial_dimension)
      .def("matrix_size", &BigWhamIOGen::MatrixSize)
      .def("get_hpattern", &BigWhamIOGen::GetHPattern)
      .def("write_hmatrix", &BigWhamIOGen::WriteHmatrix)
      .def("convert_to_global",
           [](BigWhamIOGen &self, const pbarray<double> &x) -> decltype(auto) {
             auto tx = as_array_view<double>(x);
             auto v = self.ConvertToGlobal(tx);
             return as_pyarray<double>(std::move(v));
           })
      .def("convert_to_local",
           [](BigWhamIOGen &self, const pbarray<double> &x) -> decltype(auto) {
             auto tx = as_array_view<double>(x);
             auto v = self.ConvertToLocal(tx);
             return as_pyarray<double>(std::move(v));
           })
      //.def("hdotProductInPermutted", &BigWhamIOGen::hdotProductInPermutted)
      // I change the previous binding of hdotProductInPermutted to return a
      // numpy array!!
      .def(
          "matvec_permute",
          [](BigWhamIOGen &self, const pbarray<double> &x) -> decltype(auto) {
            auto tx = as_array_view<double>(x);
            auto v = self.MatVecPerm(tx);
            return as_pyarray<double>(std::move(v));
          },
          " dot product between hmat and a vector x", py::arg("x"))

      //.def("hdotProduct",            &BigWhamIOGen::matvect, " dot product
      // between hmat and a vector x",py::arg("x"))
      // I change the previous binding of matvect to return a numpy array!!
      // todo: is it possible to move the result of the dot product to an
      // std::array? the array is contiguous in memory but not the vector!!!!!!
      // CP
      .def(
          "matvec",
          [](BigWhamIOGen &self, const pbarray<double> &x) -> decltype(auto) {
            auto tx = as_array_view<double>(x);
            auto v = self.MatVec(tx);
            return as_pyarray<double>(std::move(v));
          },
          " dot product between hmat and a vector x in original ordering",
          py::arg("x"))
      .def("get_hmat_time", &BigWhamIOGen::hmat_time)
      .def("get_omp_threads", &BigWhamIOGen::GetOmpThreads);

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

  py::class_<BigWhamIORect>(m, "BigWhamIORect", py::dynamic_attr())
      .def(py::init<>())
      .def("hmat_destructor", &BigWhamIORect::HmatrixDestructor)
      .def("set",
           py::overload_cast<
               const std::vector<double> &, const std::vector<int> &,
               const std::vector<double> &, const std::vector<int> &,
               const std::string &, const std::vector<double> &, const int,
               const double, const double>(&BigWhamIORect::Set))
      .def("set", py::overload_cast<const std::string &>(&BigWhamIORect::Set))
      .def("get_collocation_points", &BigWhamIORect::GetCollocationPoints)
      .def("get_permutation", &BigWhamIORect::GetPermutation)
      .def("get_compression_ratio", &BigWhamIORect::GetCompressionRatio)
      .def("get_kernel_name", &BigWhamIORect::kernel_name)
      .def("get_spatial_dimension", &BigWhamIORect::spatial_dimension)
      .def("matrix_size", &BigWhamIORect::MatrixSize)
      .def("get_hpattern", &BigWhamIORect::GetHPattern)
      .def("write_hmatrix", &BigWhamIOGen::WriteHmatrix)
      .def("convert_to_global",
           [](BigWhamIORect &self, const pbarray<double> &x) -> decltype(auto) {
             auto tx = as_array_view<double>(x);
             auto v = self.ConvertToGlobal(tx);
             return as_pyarray<double>(std::move(v));
           })
      .def("convert_to_local",
           [](BigWhamIORect &self, const pbarray<double> &x) -> decltype(auto) {
             auto tx = as_array_view<double>(x);
             auto v = self.ConvertToLocal(tx);
             return as_pyarray<double>(std::move(v));
           })
      .def(
          "matvec",
          [](BigWhamIORect &self, const pbarray<double> &x) -> decltype(auto) {
            auto tx = as_array_view<double>(x);
            auto v = self.MatVec(tx);
            return as_pyarray<double>(std::move(v));
          },
          " dot product between hmat and a vector x in original ordering")
      .def("get_hmat_time", &BigWhamIORect::hmat_time)
      .def("get_omp_threads", &BigWhamIORect::GetOmpThreads);
  /* --------------------------------------------------------------------------
   */

  m.def("py_get_collocation_points", &PyGetCollocationPoints);
  m.def("py_get_permutation", &PyGetPermutation);

  /* --------------------------------------------------------------------------
   */
  py::class_<Mesh>(m, "Mesh", py::dynamic_attr())
      .def(py::init([](const std::vector<double> &coor,
                       const std::vector<int> &conn,
                       const std::string &elem_type) {
        std::unique_ptr<Mesh> mesh;
        switch (hash_djb2a(elem_type)) {
        case "2DS0"_sh: {
          int spatial_dimension = 2;
          int nvertices_per_elt = 2;
          using EltType = bie::Segment<0>;
          mesh = CreateUniqueMeshFromVect<EltType>(
              spatial_dimension, nvertices_per_elt, coor, conn);
          break;
        }
        case "2DS1"_sh: {
          int spatial_dimension = 2;
          int nvertices_per_elt = 2;
          using EltType = bie::Segment<1>;
          mesh = CreateUniqueMeshFromVect<EltType>(
              spatial_dimension, nvertices_per_elt, coor, conn);
          break;
        }
        case "3DT0"_sh: {
          int spatial_dimension = 3;
          int nvertices_per_elt = 3;
          using EltType = bie::Triangle<0>;
          mesh = CreateUniqueMeshFromVect<EltType>(
              spatial_dimension, nvertices_per_elt, coor, conn);
          break;
        }
        default: {
          std::cout << "wrong inputs -abort \n";
          il::abort();
        }
        }
        return mesh;
      }))
      .def("get_collocation_points",
           [](const Mesh &self) {
             auto v = self.collocation_points();

             il::Array<double> pts{v.size(0) * v.size(1), 0.};

             int index = 0;
             for (il::int_t i = 0; i < v.size(0); i++) {
               for (il::int_t j = 0; j < v.size(1); j++) {
                 pts[index] = v(i, j);
                 index++;
               }
             }
             return as_pyarray<double>(std::move(pts));
           })
      .def("num_elements", &Mesh::num_elements)
      .def("num_collocation_points", &Mesh::num_collocation_points)
      .def("get_element_normal",
           [](const Mesh &self, const il::int_t element_id) {
             return as_pyarray<double>(
                 std::move(self.GetElement(element_id)->normal()));
           })
      .def("get_element_centroid",
           [](const Mesh &self, const il::int_t element_id) {
             return as_pyarray<double>(
                 std::move(self.GetElement(element_id)->centroid()));
           })
      .def("get_element_tangent1",
           [](const Mesh &self, const il::int_t element_id) {
             return as_pyarray<double>(
                 std::move(self.GetElement(element_id)->tangent1()));
           })
      .def("get_element_tangent2",
           [](const Mesh &self, const il::int_t element_id) {
             return as_pyarray<double>(
                 std::move(self.GetElement(element_id)->tangent2()));
           })
      .def("get_element_size",
           [](const Mesh &self, const il::int_t element_id) {
             return self.GetElement(element_id)->size();
           })
      .def("get_equivalent_eigenstrain",
           /*
             disp is in global system, 1d array: num_colloc_pts * dim
             euivalent eigenstrain
             eig = Sum (u x n + n x u) * area
             eig = eig11 eig22 eig33 eig13 eig23 eig12
           */
           [](const Mesh &self, const pbarray<double> &disp) {
             auto u = as_array_view<double>(disp);
             auto dim = self.spatial_dimension();
             il::Array<double> eig;
             if (dim == 3) {
               eig.Resize(6, 0.0);
               for (il::int_t i = 0; i < self.num_elements(); ++i) {
                 auto elem = self.GetElement(i);
                 auto num_elem_col_pts = elem->num_collocation_points();
                 auto n = elem->normal();
                 auto size = elem->size() / num_elem_col_pts ;
                 // std::cout << num_elem_col_pts << std::endl;
                 for (int j = 0; j < num_elem_col_pts; ++j) {
                   auto id = i * num_elem_col_pts * dim + j * dim;
                 // std::cout << id << std::endl;
                   eig[0] += (size * u[id + 0] * n[0]);
                   eig[1] += (size * u[id + 1] * n[1]);
                   eig[2] += (size * u[id + 2] * n[2]);
                   eig[3] += (size * 0.5 * (u[id + 0] * n[2] + u[id + 2] * n[0]));
                   eig[4] += (size * 0.5 * (u[id + 1] * n[2] + u[id + 2] * n[1]));
                   eig[5] += (size * 0.5 * (u[id + 0] * n[1] + u[id + 1] * n[0]));
                 }
               }
             }

             if (dim == 2) {
               eig.Resize(3, 0.0);
               for (il::int_t i = 0; i < self.num_elements(); ++i) {
                 auto elem = self.GetElement(i);
                 auto num_elem_col_pts = elem->num_collocation_points();
                 auto n = elem->normal();
                 auto size = elem->size() / num_elem_col_pts ;
                 for (int j = 0; j < num_elem_col_pts; ++j) {
                   auto id = i * num_elem_col_pts * dim + j * dim;
                   eig[0] += size * u[id + 0] * n[0];
                   eig[1] += size * u[id + 1] * n[1];
                   eig[2] += size * 0.5 * (u[id + 0] * n[1] + u[id + 1] * n[0]);
                 }
               }
             }

             return as_pyarray<double>(std::move(eig));
           })
      .def("get_farfield_traction",
           /*
             sigma_farfield : 6 x 1 vector
             sig = sig11 sig22 sig33 sig13 sig23 sig12
           */
           [](const Mesh &self, const pbarray<double> &sigma) {
             auto sig = as_array_view<double>(sigma);
             auto dim = self.spatial_dimension();
             il::Array<double> trac;
             trac.Resize(dim * self.num_collocation_points(), 0.0);
             if (dim == 3) {
               for (il::int_t i = 0; i < self.num_elements(); ++i) {
                 auto elem = self.GetElement(i);
                 auto num_elem_col_pts = elem->num_collocation_points();
                 auto n = elem->normal();
                 for (int j = 0; j < num_elem_col_pts; ++j) {
                   // t0
                   trac[i * num_elem_col_pts * dim + j * dim + 0] =
                       sig[0] * n[0] + sig[5] * n[1] + sig[3] * n[2];
                   // t1
                   trac[i * num_elem_col_pts * dim + j * dim + 1] =
                       sig[5] * n[0] + sig[1] * n[1] + sig[4] * n[2];
                   // t2
                   trac[i * num_elem_col_pts * dim + j * dim + 2] =
                       sig[3] * n[0] + sig[4] * n[1] + sig[2] * n[2];
                 }
               }
             }

             if (dim == 2) {
               for (il::int_t i = 0; i < self.num_elements(); ++i) {
                 auto elem = self.GetElement(i);
                 auto num_elem_col_pts = elem->num_collocation_points();
                 auto n = elem->normal();
                 for (int j = 0; j < num_elem_col_pts; ++j) {
                   // t0
                   trac[i * num_elem_col_pts * dim + j * dim + 0] =
                       sig[0] * n[0] + sig[2] * n[1];
                   // t1
                   trac[i * num_elem_col_pts * dim + j * dim + 1] =
                       sig[2] * n[0] + sig[1] * n[1];
                 }
               }
             }

             return as_pyarray<double>(std::move(trac));
           })
      .def("convert_to_global",
           [](const Mesh &self, const pbarray<double> &x) -> decltype(auto) {
             auto tx = as_array_view<double>(x);
             auto v = self.ConvertToGlobal(tx);
             return as_pyarray<double>(std::move(v));
           })
      .def("convert_to_local",
           [](const Mesh &self, const pbarray<double> &x) -> decltype(auto) {
             auto tx = as_array_view<double>(x);
             auto v = self.ConvertToLocal(tx);
             return as_pyarray<double>(std::move(v));
           });
  /* --------------------------------------------------------------------------
   */

  py::class_<BoundaryElement>(m, "BoundaryElement", py::dynamic_attr())
      .def(py::init(
          [](const std::vector<double> &verts, const std::string &elem_type) {
            std::unique_ptr<BoundaryElement> elem;
            int num_vertices;
            int spatial_dimension;

            switch (hash_djb2a(elem_type)) {
            case "2DS0"_sh: {
              spatial_dimension = 2;
              num_vertices = 2;
              using EltType = bie::Segment<0>;
              elem = std::make_unique<ElemType>();
              break;
            }
            case "2DS1"_sh: {
              spatial_dimension = 2;
              num_vertices = 2;
              using EltType = bie::Segment<1>;
              elem = std::make_unique<ElemType>();
              break;
            }
            case "3DT0"_sh: {
              spatial_dimension = 3;
              num_vertices = 3;
              using EltType = bie::Triangle<0>;
              elem = std::make_unique<ElemType>();
              break;
            }
            default: {
              std::cout << "wrong inputs -abort \n";
              il::abort();
            }
            }

            il::Array2D<double> vertices{num_vertices, spatial_dimension};
            int index = 0;
            for (il::int_t i = 0; i < num_vertices; i++) {
              for (il::int_t j = 0; j < spatial_dimension; j++) {
                vertices(i, j) = verts[index];
                index++;
              }
            }
            elem->SetElement(vertices);
            return elem;
          }))
      .def("centroid",
           [](const BoundaryElement &self) {
             return as_pyarray<double>(std::move(self.centroid()));
           })
      .def("normal",
           [](const BoundaryElement &self) {
             return as_pyarray<double>(std::move(self.normal()));
           })
      .def("tangent1",
           [](const BoundaryElement &self) {
             return as_pyarray<double>(std::move(self.tangent1()));
           })
      .def("tangent",
           [](const BoundaryElement &self) {
             return as_pyarray<double>(std::move(self.tangent2()));
           })
      .def("size", &BoundaryElement::size);
}
