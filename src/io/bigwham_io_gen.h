#ifndef BIGWHAM_IO_GEN_H
#define BIGWHAM_IO_GEN_H

#include <cstdlib>

#include "core/be_mesh.h"
#include "core/bie_kernel.h"
#include "core/elastic_properties.h"

#include "hmat/bie_matrix_generator.h"
#include "hmat/hmatrix/hmat.h"

#include "elements/point.h"
#include "elements/rectangle.h"
#include "elements/segment.h"
#include "elements/triangle.h"

#include "elasticity/bie_elastostatic.h"
#include "elasticity/fullspace_iso_axisymm_flat_ring_unidirectional/bie_elastostatic_axi3d0.h"
#include "elasticity/fullspace_iso_sp3d_segment/bie_elastostatic_sp3d.h"

using namespace bie;

class BigWhamIOGen {
private:
  std::string kernel_name_;
  il::int_t spatial_dimension_;
  il::int_t dof_dimension_;
  il::int_t max_leaf_size_;
  double eta_;
  double epsilon_aca_;
  bool is_built_;

  double h_representation_time_;
  double hmat_time_;


  std::shared_ptr<Hmat<double>> hmat_;
  std::shared_ptr<HRepresentation> hr_;
  std::shared_ptr<Mesh> mesh_src_;
  std::shared_ptr<Mesh> mesh_rec_;
  std::shared_ptr<Mesh> mesh_; // for square matrix
  std::shared_ptr<BieKernel<double>> ker_obj_;

public:
  BigWhamIOGen() {}
  ~BigWhamIOGen() {}

  // square matrices
  void SetSelf(const std::vector<double> &coor, const std::vector<int> &conn,
               const std::string &kernel, const std::vector<double> &properties,
               const int max_leaf_size, const double eta, const double eps_aca);

  // rectangular matrices
  void Set(const std::vector<double> &coor_src,
           const std::vector<int> &conn_src,
           const std::vector<double> &coor_rec,
           const std::vector<int> &conn_rec, const std::string &kernel,
           const std::vector<double> &properties, const int max_leaf_size,
           const double eta, const double eps_aca);

  void HmatDestructor();
  std::vector<double> GetCollocationPoints() const;
  std::vector<long> GetPermutation() const;
  std::vector<long> GetHPattern() const;
  void GetFullBlocks(std::vector<double> &val_list,
                     std::vector<int> &pos_list) const;
  void GetDiagonal(std::vector<double> &val_list) const;
  std::vector<double> MatVec(const std::vector<double> &x) const;
  std::vector<double> MatVecPerm(const std::vector<double> &x) const;
  std::vector<double> ConvertToGlobal(const std::vector<double> &x_local) const;
  std::vector<double> ConvertToLocal(const std::vector<double> &x_global) const;
  long MatrixSize(const int k) { return hmat_->size(k); };
  double GetCompressionRatio() const {
    IL_EXPECT_FAST(is_built_);
    return hmat_->compressionRatio();
  }
  double hmat_time() const { return hmat_time_; };
  double pattern_time() const { return h_representation_time_; };
  std::string kernel_name() const { return kernel_name_; }
  int spatial_dimension() const { return spatial_dimension_; }
  int dof_dimension() const { return dof_dimension_; }
  bool is_built() const {return is_built_;}
  void HmatrixDestructor() {
    // this function will free the memory and set the hmat obj to its initial
    // status prior to initialization this will avoid ownership specifications
    // at binding time
    this->hmat_->hmatMemFree();
  }
};

#endif // BIGWHAM_IO_GEN_H
