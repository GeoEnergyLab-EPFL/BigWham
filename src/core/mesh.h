#ifndef BIGWHAM_MESH_H
#define BIGWHAM_MESH_H

#include <iostream>
#include <memory>
#include <vector>

#include "elements/boundary_element.h"

namespace bie {

class Mesh {
protected:
  il::int_t spatial_dimension_;
  il::int_t num_coords_;
  il::int_t num_elements_;
  std::vector<std::shared_ptr<BoundaryElement>> element_list_;
  // Coordinates of the nodes - size: number of nodes x problem dimension
  il::Array2D<double> coordinates_;
  // Connectivity matrix - size: number of elements x   n_vertex per element
  il::Array2D<il::int_t> connectivity_;

  il::int_t num_collocation_points_;
  il::Array2D<double> collocation_points_;

public:
  Mesh(){};
  Mesh(const il::Array2D<double> &coordinates,
       const il::Array2D<il::int_t> &connectivity) {
    num_coords_ = coordinates.size(0);
    spatial_dimension_ = coordinates.size(1);
    num_elements_ = connectivity.size(0);
    coordinates_ = coordinates;
    connectivity_ = connectivity;
    element_list_.resize(num_elements_);
    // std::cout << element_list_.size() << std::endl;
    // std::cout << num_elements_ << std::endl;
  };
  ~Mesh(){};
  std::shared_ptr<BoundaryElement> GetElement(il::int_t element_id) const {
    return element_list_[element_id];
  };
  il::int_t num_elements() const { return num_elements_; };
  il::int_t spatial_dimension() const { return this->spatial_dimension_; };
  // nodal coordinates related.
  il::Array2D<double> coordinates() const { return coordinates_; };
  // Read a particular element of the coordinates coordinates
  double coordinates(il::int_t k, il::int_t i) { return coordinates_(k, i); }

  il::Array2D<il::int_t> connectivity() const { return connectivity_; }

  // method to return all collocation points of the mesh.
  il::Array2D<double> collocation_points() const {
    return this->collocation_points_;
  }

  il::int_t num_collocation_points() const {
    return this->num_collocation_points_;
  }

  virtual il::int_t GetElementId(il::int_t dof_index) const = 0;

  virtual il::int_t GetElementCollocationId(il::int_t dof_index) const = 0;

  // assign element list in this implementation
  virtual void ConstructMesh() = 0;

  std::vector<double>
  ConvertToGlobal(const std::vector<double> &local_vec) const;

  std::vector<double>
  ConvertToLocal(const std::vector<double> &global_vec) const;

  il::Array<double> ConvertToGlobal(il::ArrayView<double> local_vec) const;

  il::Array<double> ConvertToLocal(il::ArrayView<double> global_vec) const;
};
/* -------------------------------------------------------------------------- */

inline il::Array<double>
Mesh::ConvertToGlobal(il::ArrayView<double> local_vec) const {
  il::Array<double> v;
  IL_ASSERT(local_vec.size() == num_collocation_points_ * spatial_dimension_);
  v.Resize(local_vec.size());

#pragma omp parallel for
  for (auto i = 0; i < num_elements_; i++) {
    auto rot_mat = element_list_[i]->rotation_matrix_t();
    il::Range range{i * spatial_dimension_,
                    i * spatial_dimension_ + spatial_dimension_};
    il::blas(1.0, rot_mat.view(), local_vec.view(range), 0.0, il::io,
             v.Edit(range));
  }
  return v;
}
/* -------------------------------------------------------------------------- */
inline il::Array<double>
Mesh::ConvertToLocal(il::ArrayView<double> global_vec) const {
  il::Array<double> v;

  IL_ASSERT(global_vec.size() == num_collocation_points_ * spatial_dimension_);
  v.Resize(global_vec.size());

#pragma omp parallel for
  for (auto i = 0; i < num_elements_; i++) {
    auto rot_mat = element_list_[i]->rotation_matrix();
    il::Range range{i * spatial_dimension_,
                    i * spatial_dimension_ + spatial_dimension_};
    il::blas(1.0, rot_mat.view(), global_vec.view(range), 0.0, il::io,
             v.Edit(range));
  }

  return v;
}

/* -------------------------------------------------------------------------- */

inline std::vector<double>
Mesh::ConvertToGlobal(const std::vector<double> &local_vec) const {
  std::vector<double> v;
  IL_ASSERT(local_vec.size() == num_collocation_points_ * spatial_dimension_);
  v.resize(local_vec.size());

#pragma omp parallel for
  for (auto i = 0; i < num_elements_; i++) {
    il::Array<double> lvec_tmp;
    lvec_tmp.Resize(spatial_dimension_);
    for (int d = 0; d < spatial_dimension_; d++) {
      lvec_tmp[d] = local_vec[i * spatial_dimension_ + d];
    }
    auto v_tmp = element_list_[i]->ConvertToGlobal(lvec_tmp);
    for (int d = 0; d < spatial_dimension_; d++) {
      v[i * spatial_dimension_ + d] = v_tmp[d];
    }
  }

  return v;
}
/* -------------------------------------------------------------------------- */
inline std::vector<double>
Mesh::ConvertToLocal(const std::vector<double> &global_vec) const {
  std::vector<double> v;

  IL_ASSERT(global_vec.size() == num_collocation_points_ * spatial_dimension_);
  v.resize(global_vec.size());

#pragma omp parallel for
  for (auto i = 0; i < num_elements_; i++) {
    il::Array<double> gvec_tmp;
    gvec_tmp.Resize(spatial_dimension_);
    for (int d = 0; d < spatial_dimension_; d++) {
      gvec_tmp[d] = global_vec[i * spatial_dimension_ + d];
    }
    auto v_tmp = element_list_[i]->ConvertToLocal(gvec_tmp);
    for (int d = 0; d < spatial_dimension_; d++) {
      v[i * spatial_dimension_ + d] = v_tmp[d];
    }
  }
  return v;
}

} // namespace bie

#endif // BIGWHAM_MESH_H
