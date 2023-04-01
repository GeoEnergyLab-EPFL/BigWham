import numpy as np
from scipy.sparse.linalg import LinearOperator

from py_bigwham import BigWhamIORect
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


##############################
#  Hmatrix class in python   #
##############################
class HmatrixRectangular(LinearOperator):
    def __init__(
        self,
        kernel,
        coor_src,
        conn_src,
        coor_rec,
        conn_rec,
        properties,
        max_leaf_size=100,
        eta=3,
        eps_aca=1.0e-3,
    ):

        self.kernel_ = kernel
        self.properties_ = properties
        self.max_leaf_size_ = max_leaf_size
        self.eta_ = eta
        self.eps_aca_ = eps_aca

        self.H_ = BigWhamIORect()
        self.H_.set(
            coor_src.flatten(),
            conn_src.flatten(),
            coor_rec.flatten(),
            conn_rec.flatten(),
            kernel,
            properties.flatten(),
            max_leaf_size,
            eta,
            eps_aca,
        )
        self.matvec_size_ = self.H_.matrix_size(0)
        self.dtype_ = float

        # it is mandatory to define shape and dtype of the dot product
        self.shape_ = (self.H_.matrix_size(0), self.H_.matrix_size(1))
        super().__init__(self.dtype_, self.shape_)

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """
        This function implements the dot product.
        :param v: vector expected to be of size self.HMAT_size_
        :return: HMAT.v
        """
        return self.H_.matvec(v)

    @property
    def _init_shape(self):
        return self.shape_

    def _init_dtype(self):
        return self.dtype_

    # some useful methods
    def getCompression(self) -> float:
        return self.H_.get_compression_ratio()

    def getPermutation(self) -> np.ndarray:
        return np.asarray(self.H_.get_permutation())

    def getMeshCollocationPoints(self) -> np.ndarray:
        """
        Get collocation points from mesh (no permutations ....)
        return: (no_collo_pts, dim) array from mesh
        """
        dim = self.H_.get_spatial_dimension()
        return np.asarray(self.H_.get_collocation_points()).reshape(
            (self.matvec_size_ // dim, dim)
        )

    def convert_to_global(self, x_local: np.ndarray) -> np.ndarray:
        """
        Convert local vector to global vector
        :param x_local: local vector
        :return: global vector
        """
        return self.H_.convert_to_global(x_local)

    def convert_to_local(self, x_global: np.ndarray) -> np.ndarray:
        """
        Convert global vector to local vector
        :param x_global: global vector
        :return: local vector
        """
        return self.H_.convert_to_local(x_global)

    def getCollocationPoints(self) -> np.ndarray:
        n = self.H_.get_spatial_dimension()
        aux = np.asarray(self.H_.get_collocation_points())
        auxpermut = np.reshape(aux, (int(aux.size / n), n))
        permut = self.getPermutation()
        colPts = 0.0 * auxpermut
        colPts[permut] = auxpermut
        return colPts

    def getSpatialDimension(self) -> int:
        return self.H_.get_spatial_dimension()

    def _getPattern(self) -> np.ndarray:
        aux = np.asarray(self.H_.get_hpattern())
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
