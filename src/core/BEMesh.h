//
// This file is part of BigWham.
//
// Created by Brice Lecampion on 13.01.23.
// Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland, Geo-Energy Laboratory, 2016-2023.  All rights reserved.
// See the LICENSE.TXT file for more details.
//

#ifndef BIGWHAM_BEMESH_H
#define BIGWHAM_BEMESH_H

#pragma once

#include "core/BoundaryElement.h"

namespace bie {

// Class for Boundary element mesh
// a BE mesh has a element with dimension -1 compared to the spatial dimension...
// the coordinates of the element vertex have dimension spatial dimension
// this should replace the 2D and 3D class into a single class....
template<class E>  // E is the element type
class BEMesh {
private:

    E element_def_;  // element_def_ is an instance of the element type of the mesh.  Could be extended to an array of element ?

    il::int_t spatial_dimension_;

    il::int_t number_vertex_;  // number of vertex per element

    // Coordinates of the nodes - size: number of nodes x problem dimension
    il::Array2D<double> coordinates_;

    // Connectivity matrix - size: number of elements x   n_vertex per element
    il::Array2D<il::int_t> connectivity_;

    // Interpolation order
    il::int_t interpolation_order_;

    //  collocation points / nodes per element
    il::int_t nodes_per_element_;

    il::int_t  n_elts_; // total number of elements in the mesh

public:

    BEMesh(){};

    // Basic constructor with  coordinates and connectivity array and
    // element type !
    BEMesh(il::Array2D<double> &Coordinates, il::Array2D<il::int_t> &Connectivity) {
        //element_def_=element;
        // check validity of inputs
        IL_EXPECT_FAST(Coordinates.size(1) == element_def_.getSpatialDimension() );
        spatial_dimension_ = element_def_.getSpatialDimension();
        IL_EXPECT_FAST(spatial_dimension_ == 2 || spatial_dimension_ == 3);
        IL_EXPECT_FAST(Connectivity.size(1) == element_def_.getNumberOfVertices() );
        n_elts_ = Connectivity.size(0);
        //  we do not check consistency of Connectivity here - such that actually there can be no connectivity (can be used as dummy mesh)
        //    for list of observation points for example
        number_vertex_ = element_def_.getNumberOfVertices();
        coordinates_ = Coordinates;
        connectivity_ = Connectivity;
        interpolation_order_ = element_def_.getInterpolationOrder();
        nodes_per_element_ = element_def_.getNumberOfNodes();
    };

    //////////////////////////////////////////////////////////////////////////
    //        get-set functions  - i.e. public interfaces
    //////////////////////////////////////////////////////////////////////////

    il::int_t numberOfElts() const { return n_elts_; };
    il::int_t interpolationOrder() const { return interpolation_order_; };
    il::int_t numberOfNodes() const {return nodes_per_element_;};
    il::int_t numberCollocationPoints() const {return  (element_def_.getNumberOfCollocationPoints())*n_elts_ ;}

    // nodal coordinates related.
    il::Array2D<double> coordinates() const { return coordinates_; };
    // Read a particular element of the coordinates coordinates
    double coordinates(il::int_t k, il::int_t i)  {return coordinates_(k, i);}
    il::StaticArray<double, 2> coordinates(il::int_t k)  const
    {
        il::StaticArray<double, 2> temp;
        temp[0] = coordinates_(k, 0);
        temp[1] = coordinates_(k, 1);
        return temp;
    };
    // connectivity related
    il::Array2D<il::int_t> connectivity() const { return connectivity_; };
    // get the connectivity of an element -> A StaticArray of size 2 here !
    il::StaticArray<il::int_t, 2> connectivity(il::int_t k)  // const
    {
        il::StaticArray<il::int_t, 2> temp;
        for (il::int_t i = 0; i < connectivity_.size(1); i++) {
            temp[i] = connectivity_(k, i);
        }
        return temp;
    };
    //
    il::int_t connectivity(il::int_t e, il::int_t i)   const
    {// element e, local coordinates i - return global nodes
        return connectivity_(e, i);
    }

    il::int_t numberDDDofsPerElt() const {
        return nodes_per_element_ * spatial_dimension_;
    }

    il::int_t numberDDDofs() const {
        return (numberOfElts() * nodes_per_element_ * spatial_dimension_);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    //   Methods
    ////////////////////////////////////////////////////////////////////////////////////////////

    il::Array2D<double> getVertices(il::int_t ne) const {
        il::Array2D<double> vertElt{number_vertex_,spatial_dimension_};
        // loop over the vertices
        for (il::int_t j=0;j<spatial_dimension_;j++){
            for (il::int_t i = 0; i < number_vertex_; i++) {
                vertElt(i,j) = coordinates_(connectivity_(ne,i), j);
            }
        }
        return vertElt;
    }

    void setCurrentElement(il::int_t ne)  {
        il::Array2D<double> xv{number_vertex_,spatial_dimension_,0,};
        xv = this->getVertices(ne);
        this->element_def_.setElement(xv);
    }

    // method to return all collocation points of the mesh.
    il::Array2D<double> getCollocationPoints()  {
        il::Array2D<double> xv{number_vertex_,spatial_dimension_,0,};
        il::Array2D<double> Xcol{nodes_per_element_, spatial_dimension_, 0.},
                colPoints{n_elts_ * nodes_per_element_, spatial_dimension_, 0};
        il::int_t j = 0;
        std::cout << "number of elements: " << n_elts_  << " nodes_per el: " << nodes_per_element_  << " spatial dim: " << spatial_dimension_<<"\n";
        std::cout << "number of vertex per element: " << number_vertex_ <<"\n";
        for (il::int_t e = 0; e < n_elts_; e++) {
            this->setCurrentElement(e);
            Xcol = this->element_def_.getCollocationPoints();
            auto centro = this->element_def_.getCentroid();
          //  std::cout << "centro  " << e << "  centroid " << centro[0]  <<" -" << centro[1]   <<"\n";
           // std::cout << "xcol  " << e << "size " << Xcol(0,0) <<"-" << Xcol(0,1) <<"\n";
            for (il::int_t k=0;k<spatial_dimension_;k++){
                for (il::int_t j1 = 0; j1 < nodes_per_element_; j1++) {
                    colPoints(j+j1, k) = Xcol(j1, k);
                }
            }
            j= j + nodes_per_element_;
        };
        return colPoints;
    };

    il::int_t inElement(il::int_t point_n) const{
        return il::floor(point_n / (nodes_per_element_));  // element
    }

    il::int_t localCollocationPointId(il::int_t point_n) const {
       return (point_n % (nodes_per_element_) ) ;
    }

};

}


#endif //BIGWHAM_BEMESH_H
