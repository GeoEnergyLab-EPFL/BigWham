from hmatrix import Hmatrix  # linear operator file
import numpy as np
from timeit import default_timer as timer

# import subprocess

# subprocess.call("python generate_penny_mesh.py 0.5", shell=True)

ntimes = 1000
radius = 1.0
pressure = 1.0

G = 1.0
nu = 0.25
E = (2 * G) * (1 + nu)

max_leaf_size = 16
eta = 3.0
eps_aca = 1e-4

coord = np.load("mesh_coords.npy")
conn = np.load("mesh_conn.npy")

# Create H-matrix
kernel = "3DT0"
hmat = Hmatrix(kernel, coord, conn, np.array([E, nu]), max_leaf_size, eta, eps_aca)

col_pts = hmat.getMeshCollocationPoints()

pre_fac = (8 * (1 - nu * nu)) / (np.pi * E)
dd = np.zeros(col_pts.shape)
dd[:, 2] = pre_fac * np.sqrt(
    radius * radius - np.linalg.norm(col_pts[:, :2], axis=1) ** 2
)


# calculate tractions
start = timer()
for i in range(ntimes):
    t = hmat.matvec(dd.flatten())
end = timer()
elapsed = (end - start) / ntimes
print("Elapsed time per matvec is %f seconds." % elapsed)

t_anal = np.zeros(col_pts.shape)
t_anal[:, 2] = pressure

print("L2 Rel error {}".format(np.linalg.norm(t - t_anal.flatten()) / t.shape[0]))
