import numpy as np
import numba
from .stl import STL


def points_inside(
    stl: STL, mesh_x: np.ndarray, mesh_y: np.ndarray, mesh_z: np.ndarray, tol=1e-6
):
    """
    Check if the given points are inside or outside the volume defined by the STL
    object. This is technically called the "Point-in-Polygon" problem
    (https://en.wikipedia.org/wiki/Point_in_polygon).

    https://stackoverflow.com/questions/44513525/testing-whether-a-3d-point-is-inside-a-3d-polyhedron


    Moller-Trumbore Intersection Algorithm

    Args:

    Returns:
    """
    # We require that mesh_x, mesh_y and mesh_z all have the same shape and are 3D
    if mesh_x.shape != mesh_y.shape or mesh_x.shape != mesh_z.shape:
        raise RuntimeError("Mesh shapes are not identical")

    if mesh_x.ndim != 3:
        raise RuntimeError("Mesh must be 3D")

    return _check_points(stl.triangles, stl.bounding_box(), mesh_x, mesh_y, mesh_z, tol)


@numba.njit(parallel=True)
def _check_points(triangles, bounding_box, mesh_x, mesh_y, mesh_z, tol=1e-4):
    """
    Use a ray casting algorithm to check if the provided points (defined by the
    mesh arguments) are inside the triangulated surface or not. This function is
    JIT compiled using numba.

    This works by computing how many times a ray intersects the triangulated surface.
    If it intersects an odd number of times, the point is inside the surface.
    Otherwise, the point is outside. The intersection is computed using the
    Moller-Trumbore intersection algorithm.

    Links:
    https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm
    https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    """
    inside = np.zeros(mesh_x.shape, dtype=np.int32)
    mesh_shape = mesh_x.shape
    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounding_box
    n_triangles = triangles.shape[0]

    for i in range(mesh_shape[0]):
        for j in range(mesh_shape[1]):
            for k in numba.prange(mesh_shape[2]):
                x = mesh_x[i, j, k]
                y = mesh_y[i, j, k]
                z = mesh_z[i, j, k]
                point = np.asarray([x, y, z])

                # Initial easy test. Just check if we are inside the bounding box of
                # the triangulated surface
                if (
                    x < x_min
                    or x > x_max
                    or y < y_min
                    or y > y_max
                    or z < z_min
                    or z > z_max
                ):
                    inside[i, j, k] = 0
                    continue

                # Now we count the number of intersections to determine if the point is
                # inside the volume or not.
                intersection_count = 0
                ray =  np.random.rand(3)
                for n in range(n_triangles):
                    if _ray_intersects(triangles[n, ...], ray, point, tol):
                        intersection_count += 1

                inside[i, j, k] = intersection_count % 2

        if i % 10 == 0:
            print("Progress: ", i / mesh_shape[0] * 100, "%")

    return inside


@numba.njit
def _ray_intersects(triangle, ray, origin, tol):
    """
    Test if the given ray and triangle intersects Moller-Trumbore intersection
    algorithm. This function, and the variable names used match the code listing in
    Section 3 of the reference cited below.

    References:
    Fast Minimum Storage Ray/Triangle Intersection, Moller and Trumbore, 1997
    https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

    Returns:
        (bool): True if the given ray and triangle intersect
    """
    v0 = triangle[0, :]
    v1 = triangle[1, :]
    v2 = triangle[2, :]

    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.cross(ray, edge2)
    det = np.dot(edge1, pvec)

    # Ray is parallel to the triangle
    if np.abs(det) < tol:
        return False

    inv_det = 1.0 / det

    # Compute U parameter
    tvec = origin - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < tol or u > 1.0 - tol:
        return False

    # Compute V parameter
    qvec = np.cross(tvec, edge1)
    v = np.dot(ray, qvec) * inv_det
    if v < tol or (u + v) > 1.0 - tol:
        return False
    
    t = np.dot(edge2, qvec) * inv_det
    if t < tol:
        return False

    return True
