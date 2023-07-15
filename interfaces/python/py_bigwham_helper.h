#include "hmat/hierarchical_representation.h"
#include "il/Array.h"
#include "io/bigwham_io_gen.h"
#include "io/bigwham_io_helper.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "elasticity/fullspace_iso_2d_segment/elastic_2dP0_segment.h"

namespace py = pybind11;
template <typename T>
using pbarray = py::array_t<T, py::array::c_style | py::array::forcecast>;
/* -------------------------------------------------------------------------- */

template <typename T> inline py::array_t<T> as_pyarray(il::Array<T> &&seq) {
  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<il::Array<T>> seq_ptr =
      std::make_unique<il::Array<T>>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<il::Array<T>>(reinterpret_cast<il::Array<T> *>(p));
  });
  seq_ptr.release();
  return pbarray<T>(size, data, capsule);
}
/* -------------------------------------------------------------------------- */

template <typename T>
inline il::ArrayView<T> as_array_view(const pbarray<T> &c) {
  il::ArrayView<T> d{c.data(), c.shape(0)};
  return d;
}
/* -------------------------------------------------------------------------- */

pbarray<double> PyGetCollocationPoints(const std::vector<double> &coor,
                                       const std::vector<int> &conn,
                                       const std::string &kernel) {

  std::shared_ptr<bie::Mesh> mesh;

  switch (hash_djb2a(kernel)) {
  case "3DT0"_sh: {
    int spatial_dimension = 3;
    int nvertices_per_elt = 3;
    using EltType = bie::Triangle<0>;
    mesh = createMeshFromVect<EltType>(spatial_dimension, nvertices_per_elt,
                                       coor, conn);
    break;
  }
  default: {
    std::cout << "wrong inputs -abort \n";
    il::abort();
  }
  }
  auto v = mesh->collocation_points();

  il::Array<double> pts{v.size(0) * v.size(1), 0.};

  int index = 0;
  for (il::int_t i = 0; i < v.size(0); i++) {
    for (il::int_t j = 0; j < v.size(1); j++) {
      pts[index] = v(i, j);
      index++;
    }
  }

  return as_pyarray<double>(std::move(pts));
}
/* -------------------------------------------------------------------------- */

pbarray<il::int_t> PyGetPermutation(const int dim, const pbarray<double> &pts,
                                    const il::int_t leafsize) {

  il::Array2D<double> v{pts.size() / dim, dim, 0.};
  int index = 0;
  for (il::int_t i = 0; i < v.size(0); i++) {
    for (il::int_t j = 0; j < v.size(1); j++) {
      v(i, j) = pts.data()[index];
      index++;
    }
  }
  auto cluster = bie::cluster(leafsize, il::io, v);
  auto d = cluster.permutation;
  return as_pyarray<il::int_t>(std::move(d));
}
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

/* --------------------------------------------------------------------------
 */

/*
dim : dimension of the problem
pts : (observation points - source points) flattened
elem_kernel_type : element type and kernel type, for example "2DS0-H"
H : stress due to DD
T : stress due to point force
*/
pbarray<double> PyComputeStress(const pbarray<double> &pts,
                                const std::string &elem_kernel_type,
                                const double h = 1.0, const double G = 1.0,
                                const double nu = 0.0) {

  il::Array<double> stress;
  switch (hash_djb2a(elem_kernel_type)) {
  case "2DS0-T"_sh: {
    int dim = 2;
    int num_pts = pts.size() / dim;
    stress.Resize(num_pts * dim * 3);
    int index_pts = 0;
    int index_stress = 0;
    for (il::int_t i = 0; i < num_pts; i++) {
      auto tmp = bie::Se_segment_0(h, G, nu, pts.data()[index_pts],
                                   pts.data()[index_pts + 1]);
      // stress due to force in direction 1
      stress[index_stress] = tmp(0, 0);
      stress[index_stress + 1] = tmp(0, 1);
      stress[index_stress + 2] = tmp(0, 2);
      // stress due to force in direction 2
      stress[index_stress + 3] = tmp(1, 0);
      stress[index_stress + 4] = tmp(1, 1);
      stress[index_stress + 5] = tmp(1, 2);
      index_pts = index_pts + dim;
      index_stress = index_stress + dim * 3;
    }
    break;
  }
  case "2DS0-H"_sh: {
    int dim = 2;
    int num_pts = pts.size() / dim;
    stress.Resize(num_pts * dim * 3);
    int index_pts = 0;
    int index_stress = 0;
    for (il::int_t i = 0; i < num_pts; i++) {
      auto tmp = bie::We_segment_0(h, G, nu, pts.data()[index_pts],
                                   pts.data()[index_pts + 1]);
      // stress due to DD in direction 1 (mode II)
      stress[index_stress] = tmp(0, 0);
      stress[index_stress + 1] = tmp(0, 1);
      stress[index_stress + 2] = tmp(0, 2);
      // stress due to DD in direction 2 (mode I)
      stress[index_stress + 3] = tmp(1, 0);
      stress[index_stress + 4] = tmp(1, 1);
      stress[index_stress + 5] = tmp(1, 2);
      index_pts = index_pts + dim;
      index_stress = index_stress + dim * 3;
    }
    break;
  }
  }
  return as_pyarray<double>(std::move(stress));
}

/* --------------------------------------------------------------------------
 */

/*
dim : dimension of the problem
pts : (observation points - source points) flattened
elem_kernel_type : element type and kernel type, for example "2DS0-U"
U : displacement due to point force
*/
pbarray<double> PyComputeDisplacement(const pbarray<double> &pts,
                                      const std::string &elem_kernel_type,
                                      const double h = 1.0,
                                      const double G = 1.0,
                                      const double nu = 0.3) {

  il::Array<double> disp;
  switch (hash_djb2a(elem_kernel_type)) {
  case "2DS0-U"_sh: {
    int dim = 2;
    int num_pts = pts.size() / dim;
    disp.Resize(num_pts * dim * 2);
    int index_pts = 0;
    int index_disp = 0;
    for (il::int_t i = 0; i < num_pts; i++) {
      auto tmp = bie::Ue_segment_0(h, G, nu, pts.data()[index_pts],
                                   pts.data()[index_pts + 1]);
      // disp due to force in direction 1
      disp[index_disp] = tmp(0, 0);
      disp[index_disp + 1] = tmp(0, 1);
      // disp due to force in direction 2
      disp[index_disp + 2] = tmp(1, 0);
      disp[index_disp + 3] = tmp(1, 1);
      index_pts = index_pts + dim;
      index_disp = index_disp + dim * dim;
    }
    break;
  }
  }
  return as_pyarray<double>(std::move(disp));
}
