import sys
import pygmsh as pgm
import numpy as np


def usage():
    print("Usage:")
    print(sys.argv[0] + " h")
    exit(1)


def plot(points, triangles):
    from matplotlib import pyplot as plt

    pts = points[:, :2]
    for e in triangles:
        for idx in [[0, 1], [1, 2], [2, 0]]:
            X = pts[e[idx]]
            plt.plot(X[:, 0], X[:, 1], "-k", lw=0.1)
    plt.gca().set_aspect("equal", "datalim")
    plt.axis("off")

    plt.savefig("mesh.png", transparent=True)


##
# Triangluar mesh in radius 1 penny shape crack
##

if len(sys.argv) != 2:
    usage()

h = float(sys.argv[1])

dim = 2

with pgm.geo.Geometry() as geom:
    geom.add_circle(
        [0.0, 0.0, 0.0],
        1.0,
        mesh_size=h,
        num_sections=3,
        # If compound==False, the section borders have to be points of the
        # discretization. If using a compound circle, they don't; gmsh can
        # choose by itself where to point the circle points.
        compound=True,
    )
    # geom.add_physical(c.plane_surface, "super disk")
    mesh = geom.generate_mesh()

print("Number of points", mesh.points.shape[0])
print("Number of Elements", mesh.get_cells_type("triangle").shape[0])
plot(mesh.points, mesh.get_cells_type("triangle"))

points = np.asfortranarray(mesh.points)
triangles = np.asfortranarray(mesh.get_cells_type("triangle"))

np.save("mesh_coords", points)
np.save("mesh_conn", triangles)
