import numpy as np
import pytest

from camera_geometry.projective_space import *
from camera_geometry.vector_ops import colinear

# ---------- Point / Hyperplane basics ----------
def test_point_normalization_and_dim():
    v = np.array([3.0, 0.0, 4.0])
    p = Point(v)
    assert p.dim() == 2
    assert np.isclose(np.linalg.norm(p.vec()), 1.0, atol=1e-6)
    
def test_hyperplane_contains_true_false():
    # Plane normal π ⟂ x  -> contains(x) == True in your API (dot ≈ 0)
    pi = Hyperplane(np.array([0.0, 0.0, 1.0]))
    x_on_plane = Point(np.array([1.0, 0.0, 0.0]))
    x_off_plane = Point(np.array([0.0, 0.0, 1.0]))
    assert pi.contains(x_on_plane)
    assert not pi.contains(x_off_plane)

def test_find_plane_from_points_and_find_point_from_planes():
    # Work in P^2 (3 coords). Two independent points -> unique plane normal.
    x1 = Point(np.array([1.0, 0.0, 1.0]))
    x2 = Point(np.array([0.0, 1.0, 1.0]))
    pi = Hyperplane.find_plane([x1, x2])
    # Both points lie on the found plane
    assert pi.contains(x1)
    assert pi.contains(x2)

    # Two P² hyperplanes (lines) with independent normals -> unique intersection point.
    p1 = Hyperplane(np.array([1.0, 0.0, -1.0]))
    p2 = Hyperplane(np.array([0.0, 1.0, -1.0]))
    x = Point.find_point([p1, p2])
    assert p1.contains(x) and p2.contains(x)
    

# ---------- Projective space ----------
def test_projective_dehomogenize_and_ideal():
    P = ProjectiveSpace(dim=2)  # infinity plane is (0,0,1)
    # Affine point: last coord ≠ 0
    x = Point(np.array([2.0, -3.0, 1.0]))
    d = P.dehomogenize(x)
    # dehomogenized vector must satisfy <π, d> = 1 and same projective direction
    assert np.isclose(d[-1], 1.0, atol=1e-6)
    assert colinear(d, x.vec())
    # Ideal point (on infinity plane)
    x_inf = Point(np.array([1.0, 1.0, 0.0]))
    assert P.is_ideal_point(x_inf)
    with pytest.raises(AssertionError): 
        assert P.dehomogenize(x_inf)  # not meaningful; expect failure if attempted


# ---------- Transformations ----------
def test_transformation_apply_and_singularity_flag():
    Hm = np.array([
        [1.0, 2.0, 0.5],
        [0.0, 1.0, -0.5],
        [0.0, 0.0, 1.0],
    ])
    T = Transformation(Hm)
    inp = Point(np.array([1.0, 1.0, 1.0]))
    outp = T(inp)
    assert isinstance(outp, Point)
    assert outp.dim() == inp.dim()
    assert not T.is_singular()
    
    expected = np.array([3.5, 0.5, 1.0])
    assert np.allclose(outp.vec() / outp.vec()[-1], expected, atol=1e-9)
    
def test_composition_dimensions():
    A = Transformation(np.array([[1, 0, 0.1],
                                 [0, 1, 0.0],
                                 [0, 0, 1.0]], dtype=float))
    B = Transformation(np.array([[2, 0, 0.0],
                                 [0, 2, 0.0],
                                 [0, 0, 1.0]], dtype=float))
    C = Composition(A, B)  # should not raise
    x = Point(np.array([1.0, 2.0, 1.0]))
    y1 = A(B(x))
    y2 = C(x)
    assert colinear(y1.vec(), y2.vec())
    
# ---------- Homography ----------
def test_homography_constructor_guards_and_affinity():
    with pytest.raises(ValueError):
        Homography(np.eye(3)[:2, :3])  # not square
    with pytest.raises(ValueError):
        Homography(np.zeros((3, 3)))   # singular

    H_aff = Homography(np.array([
        [1.2, 0.0, 0.3],
        [0.0, 0.9, -0.1],
        [0.0, 0.0, 1.0],
    ]))
    assert H_aff.is_affinity()