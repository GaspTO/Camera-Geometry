import numpy as np
import pytest

from camera_geometry.euclidean_space import *
from camera_geometry.space import Composition

# ---------- Space -----------------------------------------------------

def test_euclidean_space_singleton():
    a = EuclideanSpace(3)
    b = EuclideanSpace(3)
    c = EuclideanSpace(2)

    assert a is b
    assert a is not c
    assert a.dim == 3
    assert c.dim == 2
    assert repr(a) == "R^3"


# ---------- Point -------------------------------------------------------

def test_euclidean_point_basic_and_accessors():
    x = np.array([1.0, 2.0, 3.0])
    p = EuclideanPoint(x)

    np.testing.assert_allclose(p.x, x)
    assert p.dim == 3
    assert p.space is EuclideanSpace(3)
    assert "EuclideanPoint" in repr(p)


def test_euclidean_point_equality():
    p1 = EuclideanPoint(np.array([1.0, 2.0, 3.0]))
    p2 = EuclideanPoint(np.array([1.0, 2.0, 3.0]))
    p3 = EuclideanPoint(np.array([1.0, 2.0, 3.1]))

    assert p1 == p2
    assert not (p1 == p3)


def test_euclidean_point_requires_ndarray_and_1d():
    with pytest.raises(TypeError):
        EuclideanPoint([1.0, 2.0])  # not np.ndarray

    with pytest.raises(ValueError):
        EuclideanPoint(np.zeros((2, 2)))  # not 1D


# ---------- Transformation------------------------------------------------

def test_euclidean_transformation_basic():
    A = np.array([[1.0, 2.0],
                  [0.0, 1.0]])           # 2x2
    T = EuclideanTransformation(A)

    assert T.dim == (2, 2)
    np.testing.assert_allclose(T.mat, A)

    x = EuclideanPoint(np.array([1.0, 3.0]))
    y = T(x)

    expected = np.array([7.0, 3.0])
    np.testing.assert_allclose(y.x, expected)
    assert y.space is EuclideanSpace(2)


def test_euclidean_transformation_rectangular():
    # R^3 -> R^2
    A = np.array([[1.0, 0.0, 2.0],
                  [0.0, 1.0, 3.0]])      # 2x3
    T = EuclideanTransformation(A)
    assert T.dim == (2, 3)

    x = EuclideanPoint(np.array([1.0, 2.0, 3.0]))
    y = T(x)
    expected = np.array([7.0, 11.0])
    np.testing.assert_allclose(y.x, expected)
    assert y.space is EuclideanSpace(2)


def test_euclidean_transformation_matrix_must_be_2d():
    with pytest.raises(TypeError):
        EuclideanTransformation([[1.0, 2.0], [3.0, 4.0]])  # not ndarray

    with pytest.raises(ValueError):
        EuclideanTransformation(np.array([1.0, 2.0, 3.0]))  # 1D


# ---------- Composition -------------------------------------------------------

def test_transformation_check_compatible_uses_space_identity():
    A = np.eye(3)
    T = EuclideanTransformation(A)

    p_ok = EuclideanPoint(np.array([1.0, 0.0, -1.0]))   # in R^3
    T.check_compatible(p_ok)                            # should not raise

    q_bad = EuclideanPoint(np.array([1.0, 0.0]))        # in R^2
    with pytest.raises(TypeError):
        T.check_compatible(q_bad)


def test_euclidean_transformation_composition_matches_matrix_product():
    # T2: R^3 -> R^2, T1: R^2 -> R^2
    A = np.array([[1.0, 0.0, 2.0],
                  [0.0, 1.0, 3.0]])         # 2x3
    B = np.array([[2.0, 0.0],
                  [0.0, 5.0]])              # 2x2

    T2 = EuclideanTransformation(A)
    T1 = EuclideanTransformation(B)
    comp = Composition(T1, T2)              # T1 ∘ T2 : R^3 -> R^2

    x = EuclideanPoint(np.array([1.0, 2.0, 3.0]))
    y1 = comp(x)
    y2 = EuclideanPoint(B @ (A @ x.x))

    np.testing.assert_allclose(y1.x, y2.x)
    assert y1.space is EuclideanSpace(2)
