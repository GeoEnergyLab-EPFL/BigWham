"""
 This file is part of BigWham.

 Created by Carlo Peruzzo on 12.05.21.
 Copyright (c) EPFL (Ecole Polytechnique Fédérale de Lausanne) , Switzerland,
 Geo-Energy Laboratory, 2016-2021.  All rights reserved. See the LICENSE.TXT
 file for more details.

 last modifications :: May 25, 2021

#######################################################################################
#       IMPORTANT:                                                                    #
#       to compile the binding use python >= 3.7                                      #
#       the interpreter for the .py script should be the same as compile time         #
#######################################################################################
"""

# external
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from scipy.sparse.linalg import spilu
from scipy.sparse import diags

from bigwhamPybind import Bigwhamio, pyGetFullBlocks
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


##############################
#  Hmatrix class in python   #
##############################
class Hmatrix(LinearOperator):
    def __init__(
        self, kernel, coor, conn, properties, max_leaf_size=100, eta=3, eps_aca=1.0e-3
    ):
        """ "
        Name:              Type:                Description:
        _____________      _________________    _______________________________________________
        kernel,            (string)             name of the kernel
        coor,              (numpy 2D arrray)    coordinates of the vertices of the elements (N of vert x 2)
        conn,              (numpy 2D arrray)    connectivity of the elements (N of elemts x N of vert per elem)
        properties,        (numpy 2D arrray)    list of kernel properties (Should be changed to a dict to better implement checks?)
        max_leaf_size,     (integer)            size of the largest sub block (usually 100 - 1000)
        eta,               (integer)            approximation factor (usually ~3, 0 is non compressed HMAT)
        eps_aca            (float)              approximation factor (usually 0.001 - 0.0001)
        """
        # NOTE: no specific kernel checks are implemented for now - for grown-ups only

        self.kernel_ = kernel
        self.properties_ = properties
        self.max_leaf_size_ = max_leaf_size
        self.eta_ = eta
        self.eps_aca_ = eps_aca

        self.H_ = Bigwhamio()
        self.H_.set(
            coor.flatten(),
            conn.flatten(),
            kernel,
            properties.flatten(),
            max_leaf_size,
            eta,
            eps_aca,
        )
        self.matvec_size_ = self.H_.matrixSize(0)
        self.dtype_ = float

        # it is mandatory to define shape and dtype of the dot product
        self.shape_ = (self.H_.matrixSize(0), self.H_.matrixSize(1))
        super().__init__(self.dtype_, self.shape_)

    def _matvec(self, v):
        """
        This function implements the dot product.
        :param v: vector expected to be of size self.HMAT_size_
        :return: HMAT.v
        """
        return self.H_.matvect(v)

    @property
    def _init_shape(self):
        return self.shape_

    def _init_dtype(self):
        return self.dtype_

    # some useful methods
    def getCompression(self):
        return self.H_.getCompressionRatio()

    def getPermutation(self):
        return np.asarray(self.H_.getPermutation())

    def getMeshCollocationPoints(self):
        """
        Get collocation points from mesh (no permutations ....)
        return: (no_collo_pts, dim) array from mesh
        """
        dim = self.H_.getSpatialDimension()
        return np.asarray(self.H_.get_collocation_points()).reshape(
            (self.matvec_size_ // dim, dim)
        )

    def getCollocationPoints(self):
        n = self.H_.getSpatialDimension()
        aux = np.asarray(self.H_.get_collocation_points())
        auxpermut = np.reshape(aux, (int(aux.size / n), n))
        permut = self.getPermutation()
        colPts = 0.0 * auxpermut
        colPts[permut] = auxpermut
        return colPts

    def getSpatialDimension(self):
        return self.H_.getSpatialDimension()

    def _getFullBlocks(self):
        fb = (
            pyGetFullBlocks()
        )  # not fan of this way of creating empty object and setting them after - a constructor should do something!
        fb.set(self.H_)
        val = np.asarray(fb.getValList(), dtype=float)
        col = np.asarray(fb.getColumnN(), dtype=int)
        row = np.asarray(fb.getRowN(), dtype=int)
        return csc_matrix((val, (row, col)), shape=self.shape_)

    def _getPattern(self):
        aux = np.asarray(self.H_.getHpattern())
        #  as flattened list via a pointer
        #  the numberofblocks is also returned (by reference)
        #
        #  the pattern matrix is formatted as
        # row = 1 block : i_begin,j_begin, i_end,j_end,FLAG,entry_size
        # with FLAG=0 for full rank and FLAG=1 for low rank
        #
        # we output a flatten row-major order std::vector

        nr = 6
        return np.reshape(aux, (int(aux.size / nr), nr))

    def plotPattern(self):
        data_pattern = self._getPattern()

        patches = []
        p_colors = []
        max_y = data_pattern[:, 3].max()
        for i in range(len(data_pattern)):
            height = np.abs(data_pattern[i, 0] - data_pattern[i, 2])
            width = np.abs(data_pattern[i, 1] - data_pattern[i, 3])
            y1 = max_y - data_pattern[i, 0] - height
            x1 = data_pattern[i, 1]
            rectangle = Rectangle((x1, y1), width, height)
            patches.append(rectangle)
            p_colors.append(data_pattern[i, 4])
        fig = plt.figure()
        ax = fig.add_subplot(111)

        p = PatchCollection(
            patches, cmap=matplotlib.cm.PiYG, edgecolors="black", alpha=0.4
        )
        p.set_array(p_colors)
        ax.add_collection(p)
        ax.set_ylim([data_pattern[:, 0].min(), data_pattern[:, 3].max()])
        ax.set_xlim([data_pattern[:, 1].min(), data_pattern[:, 2].max()])
        ax.set_aspect("equal")
        # fig.colorbar(p)
        fig.show()
        plt.show(block=True)
        return fig

    # a method constructing an ILU Preconditionner of the H matrix
    def H_ILU_prec(self, fill_factor=5, drop_tol=1e-5):
        # fb = self._getFullBlocks()
        fbinv = self.H_ILU(fill_factor=fill_factor, drop_tol=drop_tol)
        return LinearOperator(self.shape_, fbinv.solve)

    def H_ILU(self, fill_factor=5, drop_tol=1e-5):
        fb = self._getFullBlocks()
        fbILU = spilu(fb, fill_factor=fill_factor, drop_tol=drop_tol)
        return fbILU

    def H_diag(self):
        fb = self._getFullBlocks()
        return fb.diagonal()

    def H_jacobi_prec(self):
        diag = self.H_diag()  # return a nd.array
        overdiag = 1.0 / diag
        return diags(overdiag, dtype=float)


# --------------------------------
#
