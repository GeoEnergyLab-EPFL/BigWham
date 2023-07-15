
#ifndef BIGWHAM_IO_HELPER_H
#define BIGWHAM_IO_HELPER_H

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/be_mesh.h"

namespace bie {

/* -------------------------------------------------------------------------- */
// utilities for switch with string in C++17
// https://learnmoderncpp.com/2020/06/01/strings-as-switch-case-labels/
inline constexpr auto hash_djb2a(const std::string_view sv) {
  unsigned long hash{5381};
  for (unsigned char c : sv) {
    hash = ((hash << 5) + hash) ^ c;
  }
  return hash;
}
/* -------------------------------------------------------------------------- */

inline constexpr auto operator"" _sh(const char *str, size_t len) {
  return hash_djb2a(std::string_view{str, len});
}
/* -------------------------------------------------------------------------- */

template <class El>
std::shared_ptr<Mesh> createMeshFromVect(int spatial_dimension,
                                         int n_vertex_elt,
                                         const std::vector<double> &coor,
                                         const std::vector<int> &conn) {
  il::int_t npoints = coor.size() / spatial_dimension;
  il::int_t nelts = conn.size() / n_vertex_elt;
  il::Array2D<double> Coor{npoints, spatial_dimension, 0.}; //
  il::Array2D<il::int_t> Conn{nelts, n_vertex_elt, 0};
  // std::cout << "in create mesh - start "
  //           << "\n";
  // std::cout << "n elems" << nelts << "\n";
  // populate mesh  ...
  int index = 0;
  for (il::int_t i = 0; i < Coor.size(0); i++) {
    for (il::int_t j = 0; j < Coor.size(1); j++) {
      Coor(i, j) = coor[index];
      index++;
    }
  }
  index = 0;
  for (il::int_t i = 0; i < Conn.size(0); i++) {
    for (il::int_t j = 0; j < Conn.size(1); j++) {
      Conn(i, j) = conn[index];
      index++;
    }
  }
  // BEMesh<El> mesh(Coor, Conn);
  auto mesh = std::make_shared<BEMesh<El>>(Coor, Conn);
  mesh->ConstructMesh();
  // std::cout << "in create mesh - done "
  //           << "\n";
  return mesh;
}
/* -------------------------------------------------------------------------- */

template <class El>
std::unique_ptr<Mesh> CreateUniqueMeshFromVect(int spatial_dimension,
                                         int n_vertex_elt,
                                         const std::vector<double> &coor,
                                         const std::vector<int> &conn) {
  il::int_t npoints = coor.size() / spatial_dimension;
  il::int_t nelts = conn.size() / n_vertex_elt;
  il::Array2D<double> Coor{npoints, spatial_dimension, 0.}; //
  il::Array2D<il::int_t> Conn{nelts, n_vertex_elt, 0};
  // populate mesh  ...
  int index = 0;
  for (il::int_t i = 0; i < Coor.size(0); i++) {
    for (il::int_t j = 0; j < Coor.size(1); j++) {
      Coor(i, j) = coor[index];
      index++;
    }
  }
  index = 0;
  for (il::int_t i = 0; i < Conn.size(0); i++) {
    for (il::int_t j = 0; j < Conn.size(1); j++) {
      Conn(i, j) = conn[index];
      index++;
    }
  }
  // BEMesh<El> mesh(Coor, Conn);
  auto mesh = std::make_unique<BEMesh<El>>(Coor, Conn);
  mesh->ConstructMesh();
  // std::cout << "in create mesh - done "
  //           << "\n";
  return mesh;
}
/* -------------------------------------------------------------------------- */

} // namespace bie

#endif // BIGWHAM_IO_HELPER_H
