""" Triangulation functions. """
import numpy as np
from numba import njit
from scipy.spatial import Delaunay


@njit
def triangle_rasterization(points, width=1):
    """ Transform triangle to surface of the fixed thickness.

    Parameters
    ----------
    points : numpy.ndarray
        array of size 3 x 3: each row is a vertex of triangle
    width : int
        thicc

    Return
    ------
    numpy.ndarray
        array of size N x 3 where N is a number of points in rasterization.
    """
    max_n_points = np.int32(triangle_volume(points, width))
    _points = np.empty((max_n_points, 3))
    i = 0
    r_margin = width - width // 2
    l_margin = width // 2
    for x in range(int(np.min(points[:, 0]))-l_margin, int(np.max(points[:, 0]))+r_margin): # pylint: disable=not-an-iterable
        for y in range(int(np.min(points[:, 1]))-l_margin, int(np.max(points[:, 1])+r_margin)):
            for z in range(int(np.min(points[:, 2]))-l_margin, int(np.max(points[:, 2]))+r_margin):
                node = np.array([x, y, z])
                if distance_to_triangle(points, node) <= width / 2:
                    _points[i] = node
                    i += 1
    return _points[:i]

@njit
def triangle_volume(points, width):
    """ Compute triangle volume to estimate the number of points. """
    a = points[0] - points[1]
    a = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

    b = points[0] - points[2]
    b = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

    c = points[2] - points[1]
    c = np.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)

    p = (a + b + c) / 2
    S = (p * (p - a) * (p - b) * (p - c)) ** 0.5
    r = S / p
    r_ = r + width + 1
    p_ = p * r_ / r
    return p_ * r_ * (width + 1)

def sticks_to_simplices(sticks, orientation, max_simplices_depth=None, max_nodes_distance=None):
    """ Compute triangulation of the fault.

    Parameters
    ----------
    sticks : numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.

    Return
    ------
    simplices : numpy.ndarray
        Array of simplices where each item is a sequence of 3 nodes indices in initial flatten array.
    nodes : numpy.ndarray
        Concatenated array of sticks nodes.
    """
    if len(sticks) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if len(sticks) == 1:
        return np.zeros((0, 3)), sticks[0]
    all_simplices = []
    nodes = np.concatenate(sticks)
    shift = 0
    for s1, s2 in zip(sticks[:-1], sticks[1:]):
        simplices = connect_two_sticks(s1, s2, orientation=orientation, max_nodes_distance=max_nodes_distance)
        if len(simplices) > 0:
            simplices += shift
            all_simplices.append(simplices)
        shift += len(s1)
    if len(all_simplices) > 0:
        all_simplices = np.concatenate(all_simplices)
        mask = filter_triangles(all_simplices, nodes, max_simplices_depth)
        return all_simplices[mask], nodes
    return np.zeros((0, 3)), np.zeros((0, 3))

def connect_two_sticks(nodes1, nodes2, axis=2, orientation=0, max_nodes_distance=20):
    """ Create triangles for two sequential sticks. """
    ranges1, ranges2 = filter_points(nodes1, nodes2, axis, max_nodes_distance)

    p1, p2 = nodes1[slice(*ranges1)], nodes2[slice(*ranges2)]

    points = np.concatenate([p1, p2])
    if len(points) <= 3:
        return []
    try:
        simplices = Delaunay(points[:, [orientation, axis]]).simplices
    except: # pylint: disable=bare-except
        return []
    l1 = (ranges1[1] - ranges1[0])
    simplices[simplices >= l1] += ranges2[0] + (len(nodes1) - ranges1[1])
    simplices += ranges1[0]
    return simplices

def filter_points(nodes1, nodes2, axis=2, max_nodes_distance=20):
    """ Remove nodes which are too far from each other. """
    if max_nodes_distance is None:
        return (0, len(nodes1)), (0, len(nodes2))

    swap = False
    if nodes2[0, axis] < nodes1[0, axis]:
        swap = True
        nodes1, nodes2 = nodes2, nodes1

    for start in range(len(nodes1)):
        if (nodes1[start, axis] - nodes2[0, axis]) > -max_nodes_distance:
            break

    ranges1 = start, len(nodes1)
    ranges2 = 0, len(nodes2)

    if swap:
        nodes1, nodes2 = nodes2, nodes1
        ranges1, ranges2 = ranges2, ranges1

    swap = False
    if nodes2[-1, axis] < nodes1[-1, axis]:
        swap = True
        nodes1, nodes2 = nodes2, nodes1
        ranges1, ranges2 = ranges2, ranges1

    for end in range(len(nodes2)-1, -1, -1):
        if (nodes2[end, axis] - nodes1[-1, axis]) < max_nodes_distance:
            break

        ranges2 = ranges2[0], end

    if swap:
        nodes1, nodes2 = nodes2, nodes1
        ranges1, ranges2 = ranges2, ranges1

    return ranges1, ranges2

def filter_triangles(triangles, points, max_simplices_depth=10):
    """ Remove large triangles. """
    mask = np.ones(len(triangles), dtype='bool')
    if max_simplices_depth is not None:
        for i, tri in enumerate(triangles):
            if points[tri].ptp(axis=0).max() > max_simplices_depth:
                mask[i] = 0
    return mask

@njit
def distance_to_triangle(triangle, node):
    """ Paper: https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    Realization: https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e """
    # pylint: disable=invalid-name, too-many-nested-blocks, too-many-branches, too-many-statements
    B = triangle[0, :]
    E0 = triangle[1, :] - B
    E1 = triangle[2, :] - B
    D = B - node
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if det == 0:
        return 0.

    # Terrible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    sqrdistance = max(sqrdistance, 0)
    dist = np.sqrt(sqrdistance)
    return dist
