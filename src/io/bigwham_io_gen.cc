#include "bigwham_io_gen.h"
#include "bigwham_io_helper.h"
#include "hmat/hierarchical_representation.h"
#include <memory>
/* -------------------------------------------------------------------------- */

// for rectangular matrix
void BigWhamIOGen::Set(
    const std::vector<double> &coor_src, const std::vector<int> &conn_src,
    const std::vector<double> &coor_rec, const std::vector<int> &conn_rec,
    const std::string &kernel, const std::vector<double> &properties,
    const int max_leaf_size, const double eta, const double eps_aca) {

  this->kernel_name_ = kernel;
  max_leaf_size_ = max_leaf_size;
  eta_ = eta;
  epsilon_aca_ = eps_aca;
  ElasticProperties elas(properties[0], properties[1]);
  std::cout << " Now setting things for kernel ... " << kernel_name_
            << " with properties size " << properties.size() << "\n";
  il::Timer tt;

  switch (hash_djb2a(kernel_name_)) {
  case "2DS0-2DP-T"_sh: {
    // 2D Segment0 and 2D Point
    spatial_dimension_ = 2;
    using src_elem = Segment<0>;
    using rec_elem = Point<2>;
    mesh_src_ = createMeshFromVect<src_elem>(
        spatial_dimension_, /* num vertices */ 2, coor_src, conn_src);
    mesh_rec_ = createMeshFromVect<rec_elem>(
        spatial_dimension_, /* num vertices */ 1, coor_src, conn_src);
    ker_obj_ = std::make_shared<
        BieElastostatic<src_elem, rec_elem, ElasticKernelType::T>>(
        elas, spatial_dimension_);
    break;
  }
  default: {
    std::cout << "wrong inputs -abort \n";
    il::abort();
  }
  }
  mesh_ = mesh_src_;
  tt.Start();
  this->hr_ = HRepresentationRectangularMatrix(mesh_src_, mesh_rec_,
                                               max_leaf_size_, eta_);
  tt.Stop();
  h_representation_time_ = tt.time();
  tt.Reset();
  tt.Start();
  bie::BieMatrixGenerator<double> M(mesh_src_, mesh_rec_, ker_obj_, hr_);
  hmat_ = std::make_shared<Hmat<double>>(M, epsilon_aca_);
  tt.Stop();
  hmat_time_ = tt.time();
  if (hmat_->isBuilt()) {
    is_built_ = true;
    dof_dimension_ = hmat_->dofDimension();
    std::cout << "HMAT --> built \n";
    double test_cr = hmat_->compressionRatio();
    std::cout << "HMAT set"
              << ", CR = " << test_cr << ", eps_aca = " << epsilon_aca_
              << ", eta = " << eta_ << "\n";
    // std::cout << "H-mat construction time = :  " << hmat_time_ << "\n";
  } else {
    is_built_ = false;
  }

  std::cout << "BigWhamIO ENDED\n";
}
/* -------------------------------------------------------------------------- */

// for square matrix
// coor and conn are assumed to be passed in row-major storage format
void BigWhamIOGen::SetSelf(const std::vector<double> &coor,
                           const std::vector<int> &conn,
                           const std::string &kernel,
                           const std::vector<double> &properties,
                           const int max_leaf_size, const double eta,
                           const double eps_aca) {
  this->kernel_name_ = kernel;
  max_leaf_size_ = max_leaf_size;
  eta_ = eta;
  epsilon_aca_ = eps_aca;
  ElasticProperties elas(properties[0], properties[1]);
  std::cout << " Now setting things for kernel ... " << kernel_name_
            << " with properties size " << properties.size() << "\n";
  il::Timer tt;

  // switch depending on Kernels for mesh building
  if (kernel_name_ == "S3DP0") {
    IL_ASSERT(properties.size() == 3);
  } else {
    IL_ASSERT(properties.size() == 2);
  }
  switch (hash_djb2a(kernel_name_)) {
  case "2DP0"_sh: {
    spatial_dimension_ = 2;
    int nvertices_per_elt_ = 2;
    using EltType = Segment<0>;
    mesh_ = createMeshFromVect<EltType>(spatial_dimension_, nvertices_per_elt_,
                                        coor, conn);
    ker_obj_ = std::make_shared<
        bie::BieElastostatic<EltType, EltType, bie::ElasticKernelType::H>>(
        elas, spatial_dimension_);
    break;
  }
  case "S3DP0"_sh: {
    spatial_dimension_ = 2;
    int nvertices_per_elt_ = 2;
    using EltType = bie::Segment<0>;
    mesh_ = createMeshFromVect<EltType>(spatial_dimension_, nvertices_per_elt_,
                                        coor, conn);
    ker_obj_ = std::make_shared<
        bie::BieElastostaticSp3d<EltType, EltType, bie::ElasticKernelType::H>>(
        elas, spatial_dimension_);

    il::Array<double> prop{1, properties[2]};
    ker_obj_->set_kernel_properties(prop);
    break;
  }
  case "2DP1"_sh: {
    spatial_dimension_ = 2;
    int nvertices_per_elt_ = 2;
    using EltType = bie::Segment<1>;
    mesh_ = createMeshFromVect<EltType>(spatial_dimension_, nvertices_per_elt_,
                                        coor, conn);
    ker_obj_ = std::make_shared<
        bie::BieElastostatic<EltType, EltType, bie::ElasticKernelType::H>>(
        elas, spatial_dimension_);
    break;
  }
  case "3DT0"_sh: {
    spatial_dimension_ = 3;
    int nvertices_per_elt_ = 3;
    using EltType = bie::Triangle<0>;
    mesh_ = createMeshFromVect<EltType>(spatial_dimension_, nvertices_per_elt_,
                                        coor, conn);
    ker_obj_ = std::make_shared<
        bie::BieElastostatic<EltType, EltType, bie::ElasticKernelType::H>>(
        elas, spatial_dimension_);
    break;
  }
  case "Axi3DP0"_sh: {
    spatial_dimension_ = 2;
    int nvertices_per_elt_ = 2;
    using EltType = bie::Segment<0>;
    mesh_ = createMeshFromVect<EltType>(spatial_dimension_, nvertices_per_elt_,
                                        coor, conn);
    ker_obj_ = std::make_shared<bie::ElasticAxiSymmRingKernel>(
        elas, spatial_dimension_);
    break;
  }
  default: {
    std::cout << "wrong inputs -abort \n";
    il::abort();
  }
  }
  tt.Start();
  this->hr_ = HRepresentationSquareMatrix(mesh_, max_leaf_size, eta);
  tt.Stop();
  h_representation_time_ = tt.time();
  tt.Reset();
  tt.Start();
  bie::BieMatrixGenerator<double> M(mesh_, mesh_, ker_obj_, hr_);
  hmat_ = std::make_shared<Hmat<double>>(M, epsilon_aca_);
  tt.Stop();
  hmat_time_ = tt.time();
  if (hmat_->isBuilt()) {
    is_built_ = true;
    dof_dimension_ = hmat_->dofDimension();
    std::cout << "HMAT --> built \n";
    double test_cr = hmat_->compressionRatio();
    std::cout << "HMAT set"
              << ", CR = " << test_cr << ", eps_aca = " << epsilon_aca_
              << ", eta = " << eta_ << "\n";
    // std::cout << "H-mat construction time = :  " << hmat_time_ << "\n";
  } else {
    is_built_ = false;
  }

  std::cout << "BigWhamIO ENDED\n";
}
/* --------------------------------------------------------------------------*/

std::vector<long> BigWhamIOGen::GetHPattern() const {
  // API function to output the hmatrix pattern
  //  as flattened list via a pointer
  //  the numberofblocks is also returned (by reference)
  //
  //  the pattern matrix is formatted as
  // row = 1 block : i_begin,j_begin, i_end,j_end,FLAG,entry_size
  // with FLAG=0 for full rank and FLAG=1 for low rank
  //
  // we output a flatten row-major order std::vector

  IL_EXPECT_FAST(is_built_);

  bie::HPattern pattern = hmat_->pattern();

  long numberofblocks = pattern.n_B;
  long len = 6 * numberofblocks;
  std::cout << "number of blocks " << numberofblocks << "\n";

  std::vector<long> patternlist(len, 0);

  int index = 0;
  //  starts with full rank
  for (il::int_t j = 0; j < pattern.n_FRB; j++) {
    // check is low rank or not
    //  il::Array2DView<double> A = hmat_.asFullRank(s);
    patternlist[index++] = pattern.FRB_pattern(1, j);
    patternlist[index++] = pattern.FRB_pattern(2, j);
    patternlist[index++] = pattern.FRB_pattern(3, j);
    patternlist[index++] = pattern.FRB_pattern(4, j);
    patternlist[index++] = 0;
    patternlist[index++] =
        (pattern.FRB_pattern(4, j) - pattern.FRB_pattern(2, j)) *
        (pattern.FRB_pattern(3, j) -
         pattern.FRB_pattern(1, j)); // size of that sub blocks
  }
  // then low ranks
  for (il::int_t j = 0; j < pattern.n_LRB; j++) {

    patternlist[index++] = pattern.LRB_pattern(1, j);
    patternlist[index++] = pattern.LRB_pattern(2, j);
    patternlist[index++] = pattern.LRB_pattern(3, j);
    patternlist[index++] = pattern.LRB_pattern(4, j);
    patternlist[index++] = 1;
    patternlist[index++] = pattern.LRB_pattern(5, j); // the rank
  }
  // return a row major flatten vector
  return patternlist;
}
/* --------------------------------------------------------------------------*/

void BigWhamIOGen::GetFullBlocks(std::vector<double> &val_list,
                                 std::vector<int> &pos_list) const {
  // return the full dense block entries of the hmat as
  // flattened lists
  // val_list(i) = H(pos_list(2*i),pos_list(2*i+1));
  // output in the original dof state (accounting for the permutation)

  IL_EXPECT_FAST(is_built_);

  il::Array<double> values{};
  il::Array<int> positions{};
  hmat_->fullBlocksOriginal(il::io, values, positions);
  //    std::cout << " checking values size" << values.size() <<  "\n";
  val_list.reserve(values.size());
  for (il::int_t i = 0; i < values.size(); i++) {
    val_list.push_back(values[i]);
  }
  pos_list.reserve(positions.size());
  for (il::int_t i = 0; i < positions.size(); i++) {
    pos_list.push_back(positions[i]);
  }
  std::cout << "number of entries " << val_list.size() << " - "
            << pos_list.size() << "\n";
  std::cout << " End of Bigwhamio getFullBlocks \n";
}
/* --------------------------------------------------------------------------*/

void BigWhamIOGen::GetDiagonal(std::vector<double> &val_list) const {
  // return the diagonal of the h-matrix
  // output in the original dof state (accounting for the permutation)

  IL_EXPECT_FAST(is_built_);
  val_list = hmat_->diagonalOriginal();

  std::cout << " End of Bigwhamio getDiagonal() \n";
}
/* -------------------------------------------------------------------------- */

std::vector<double> BigWhamIOGen::MatVec(const std::vector<double> &x) const {
  // in the original / natural ordering
  IL_EXPECT_FAST(this->is_built_);
  IL_EXPECT_FAST(hmat_->size(0) == hmat_->size(1));
  IL_EXPECT_FAST(hmat_->size(1) == x.size());
  std::vector<double> y = hmat_->matvecOriginal(x);
  return y;
}
/* -------------------------------------------------------------------------- */

std::vector<double>
BigWhamIOGen::MatVecPerm(const std::vector<double> &x) const {
  // in the permutted state.
  IL_EXPECT_FAST(this->is_built_);
  IL_EXPECT_FAST(hmat_->size(0) == hmat_->size(1));
  IL_EXPECT_FAST(hmat_->size(1) == x.size());
  std::vector<double> y = hmat_->matvec(x);
  return y;
}
/* -------------------------------------------------------------------------- */

std::vector<double>
BigWhamIOGen::ConvertToGlobal(const std::vector<double> &x_local) const {
  // Input: x in original state (not permutted)
  // Output: in original state (not permutted)
  return mesh_->ConvertToGlobal(x_local);
}
/* -------------------------------------------------------------------------- */

std::vector<double>
BigWhamIOGen::ConvertToLocal(const std::vector<double> &x_global) const {
  // Input: x in original state (not permutted)
  // Output: in original state (not permutted)
  return mesh_->ConvertToLocal(x_global);
}
/* -------------------------------------------------------------------------- */

std::vector<double> BigWhamIOGen::GetCollocationPoints() const {
  IL_EXPECT_FAST(is_built_);

  auto col_pts = mesh_->collocation_points();
  IL_EXPECT_FAST(col_pts.size(1) == spatial_dimension_);
  il::int_t npoints = col_pts.size(0);
  std::vector<double> flat_col;
  flat_col.assign(npoints * spatial_dimension_, 0.);
  int index = 0;
  for (il::int_t i = 0; i < col_pts.size(0); i++) {
    for (il::int_t j = 0; j < col_pts.size(1); j++) {
      flat_col[index] = col_pts(i, j);
      index++;
    }
  }
  return flat_col;
}
/* -------------------------------------------------------------------------- */

std::vector<long> BigWhamIOGen::GetPermutation() const {
  IL_EXPECT_FAST(is_built_);
  std::vector<long> permut;
  permut.assign(hr_->permutation_1_.size(), 0);
  for (il::int_t i = 0; i < hr_->permutation_1_.size(); i++) {
    permut[i] = hr_->permutation_1_[i];
  }
  return permut;
}
/* -------------------------------------------------------------------------- */
