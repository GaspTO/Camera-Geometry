import numpy as np
import pytest 

from camera_geometry.space import Space, Element, Transformation, Composition
from camera_geometry.projective_space import *


# ---------- ProjectiveSpace ---------------------------------------------------

def test_projective_space_singleton():
    a = ProjectiveSpace(2)
    b = ProjectiveSpace(2)
    c = ProjectiveSpace(3)  
    assert a is b
    assert a is not c
    assert a.dim == 2 and c.dim == 3
    assert repr(a) == "P^2"


# ---------- ProjectivePoint ---------------------------------------------------

def test_projective_point_basic_and_accessors():
    p = ProjectivePoint([1.0, 2.0, 3.0])
    assert p.dim == 2
    np.testing.assert_allclose(p.h, np.array([1.0, 2.0, 3.0]))

def test_projective_point_h_coord_and_unit():
    p = ProjectivePoint([1.0, 2.0, 3.0])
    h_last = p.h_coord()           # default last coord -> 1
    np.testing.assert_allclose(h_last, np.array([1/3, 2/3, 1.0]))
    h0 = p.h_coord(i=1)
    np.testing.assert_allclose(h0, np.array([1/2, 1, 3/2]))
    # unit-norm is just a representative (sign ambiguity remains)
    hun = p.h_unit()
    assert np.isclose(np.linalg.norm(hun), 1.0)

def test_projective_point_chart_exception_for_infinity():
    p = ProjectivePoint([-1.0, -2.0, 0.0])
    with pytest.raises(ValueError):
        _ = p.h_coord()  # last coord is zero → not in this chart
    # but other charts work
    np.testing.assert_allclose(p.h_coord(i=0), np.array([1.0, 2.0, 0.0]))

def test_projective_point_equals_up_to_scale():
    p = ProjectivePoint([1.0, 2.0, 3.0])
    q = ProjectivePoint([2.0, 4.0, 6.0])
    r = ProjectivePoint([1.0, 2.1, 3.0])
    assert p == q
    assert p.equals_up_to_scale(q)
    assert not p.equals_up_to_scale(r)

def test_projective_point_unit_not_sufficient_for_equality():
    # Illustrates why equals_up_to_scale can't rely on h_unit()
    p = ProjectivePoint([1.0, 0.0, 0.0])
    q = ProjectivePoint([-1.0, 0.0, 0.0])
    assert p == q                          # same projective point
    assert not np.allclose(p.h_unit(), q.h_unit())  # unit reps differ by sign


# ---------- ProjectiveTransformation -----------------------------------------

def test_projective_transformation_square_homography():
    # P^2 -> P^2, diagonal matrix (full rank)
    A = np.diag([2.0, 3.0, 4.0])
    T = ProjectiveTransformation(A)
    assert T.dim == (2, 2)
    x = ProjectivePoint([1.0, 2.0, 1.0])         # x ∈ P^2
    y = T(x)
    # y should be [2, 6, 4] ~ projectively
    np.testing.assert_allclose(y.h_coord(), np.array([0.5, 1.5, 1.0]))  # divide by last coord
    assert y.space is ProjectiveSpace(2)

def test_projective_transformation_rectangular_m_ge_n():
    # P^2 -> P^3 via (4 x 3) full-column-rank matrix
    A = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                  [1., 1., 1.]])
    assert np.linalg.matrix_rank(A) == 3
    T = ProjectiveTransformation(A)
    assert T.dim == (3, 2)
    x = ProjectivePoint([1.0, 2.0, 3.0])
    y = T(x)
    np.testing.assert_allclose(A @ x.h, y.h)

def test_projective_transformation_rejects_low_rank():
    # Rank < n+1 should raise
    A = np.array([[1., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]])  # rank 1, n+1 would be 3
    with pytest.raises(ValueError):
        _ = ProjectiveTransformation(A)

def test_transformation_check_compatible_uses_space_identity():
    A = np.eye(3)  # P^2 -> P^2
    T = ProjectiveTransformation(A)
    p_ok = ProjectivePoint([1.0, 0.0, 1.0])              # in P^2
    T.check_compatible(p_ok)                              # should not raise
    # Create a point in a different space (P^3), same "dim" logic but different instance
    q_bad = ProjectivePoint([1.0, 0.0, 0.0, 1.0])         # in P^3
    with pytest.raises(TypeError):
        T.check_compatible(q_bad)

def test_composition_matches_matrix_product():
    # t2: P^2 -> P^3 (4x3), t1: P^3 -> P^3 (4x4)
    A = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                  [1., 1., 1.]])         # rank 3
    H = np.array([[2., 0., 0., 0.],
                  [0., 3., 0., 0.],
                  [0., 0., 4., 0.],
                  [0., 0., 0., 5.]])     # invertible 4x4
    t2 = ProjectiveTransformation(A)      # P^2 -> P^3
    t1 = ProjectiveTransformation(H)      # P^3 -> P^3
    comp = Composition(t1, t2)            # t1 ∘ t2 : P^2 -> P^3

    x = ProjectivePoint([1.0, 2.0, 3.0])
    y1 = comp(x)
    y2 = ProjectivePoint(H @ (A @ x.h))
    assert y1.equals_up_to_scale(y2)


# ---------- DualProjectiveSpace ----------------------------------------------

def test_dual_projective_space_singleton():
    a = DualProjectiveSpace(2)
    b = DualProjectiveSpace(2)
    c = DualProjectiveSpace(3)
    assert a is b
    assert a is not c
    assert a.dim == 2 and c.dim == 3
    assert repr(a) == "(P^2)*"


# ---------- ProjectiveHyperplane ---------------------------------------------

def test_projective_hyperplane_basic_and_accessors():
    H = ProjectiveHyperplane([1.0, 2.0, 3.0])
    assert H.dim == 2
    np.testing.assert_allclose(H.a, np.array([1.0, 2.0, 3.0]))

def test_projective_hyperplane_a_coord_and_unit():
    H = ProjectiveHyperplane([1.0, 2.0, 3.0])
    a_last = H.a_coord()              # default last coord -> 1
    np.testing.assert_allclose(a_last, np.array([1/3, 2/3, 1.0]))
    a1 = H.a_coord(i=1)
    np.testing.assert_allclose(a1, np.array([0.5, 1.0, 1.5]))
    a_unit = H.a_unit()
    assert np.isclose(np.linalg.norm(a_unit), 1.0)

def test_projective_hyperplane_equals_up_to_scale():
    H1 = ProjectiveHyperplane([1.0, 2.0, 3.0])
    H2 = ProjectiveHyperplane([2.0, 4.0, 6.0])
    H3 = ProjectiveHyperplane([1.0, 2.1, 3.0])
    assert H1 == H2
    assert H1.equals_up_to_scale(H2)
    assert not H1.equals_up_to_scale(H3)


# ---------- Incidence --------------------------------------------------------

def test_incidence_true_and_false():
    # line z = 0 in P^2  → a = (0, 0, 1)
    H = ProjectiveHyperplane([0.0, 0.0, 1.0])
    p_on = ProjectivePoint([1.0, 2.0, 0.0])
    p_off = ProjectivePoint([1.0, 2.0, 3.0])

    assert incidence(p_on, H)
    assert not incidence(p_off, H)


# ---------- Meet / Join ------------------------------------------------------

def test_meet_unique_two_coordinate_lines_in_P2():
    # Hyperplanes x = 0 and y = 0 in P^2
    Hx = ProjectiveHyperplane([1.0, 0.0, 0.0])
    Hy = ProjectiveHyperplane([0.0, 1.0, 0.0])

    p = meet_unique([Hx, Hy])
    # intersection should be [0, 0, 1] up to scale
    q = ProjectivePoint([0.0, 0.0, 1.0])
    assert p.equals_up_to_scale(q)

def test_join_hyperplane_two_points_infinity_line_P2():
    # Points on the line z = 0 in P^2
    p1 = ProjectivePoint([1.0, 0.0, 0.0])
    p2 = ProjectivePoint([0.0, 1.0, 0.0])

    H = join_hyperplane([p1, p2])
    expected = ProjectiveHyperplane([0.0, 0.0, 1.0])  # z = 0
    assert H.equals_up_to_scale(expected)


# ---------- Dehomogenize -----------------------------------------------------

def test_dehomogenize_with_default_infinity_plane():
    # default infinity: x_n = 0 → a = e_n
    p = ProjectivePoint([1.0, 2.0, 3.0])
    x_prime = dehomogenize(p)  # should scale so last coord = 1
    np.testing.assert_allclose(x_prime, np.array([1/3, 2/3, 1.0]))

def test_dehomogenize_with_custom_infinity_plane():
    # infinity plane: x + y + z = 0 → a = (1,1,1)
    H_inf = ProjectiveHyperplane([1.0, 1.0, 1.0])
    p = ProjectivePoint([1.0, 0.0, 1.0])  # a·h = 2
    x_prime = dehomogenize(p, infinity_plane=H_inf)
    # should scale by 1/2: [0.5, 0.0, 0.5] and satisfy a^T x' = 1
    np.testing.assert_allclose(x_prime, np.array([0.5, 0.0, 0.5]))
    assert np.isclose(float(np.dot(H_inf.a, x_prime)), 1.0, atol=1e-12)

def test_dehomogenize_raises_for_point_on_infinity_plane():
    # infinity plane: z = 0
    H_inf = ProjectiveHyperplane([0.0, 0.0, 1.0])
    p = ProjectivePoint([1.0, 2.0, 0.0])  # lies on z=0
    with pytest.raises(ValueError):
        _ = dehomogenize(p, infinity_plane=H_inf)
