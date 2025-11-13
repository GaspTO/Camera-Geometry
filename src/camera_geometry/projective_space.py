from __future__ import annotations
import numpy as np
from scipy.linalg import null_space
from typing import Tuple, Iterable
from .space import Space, Transformation, Element


class ProjectiveSpace(Space):
    """Projective space P^n (over R in this project)."""

    _cache: dict[int, "ProjectiveSpace"] = {}

    def __new__(cls, dim: int):
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("dim must be a non-negative int")
        inst = cls._cache.get(dim)
        if inst is not None:
            return inst
        inst = super().__new__(cls)
        cls._cache[dim] = inst
        return inst

    def __init__(self, dim: int):
        # idempotent init (since __init__ runs even when returning a cached instance)
        if getattr(self, "_init_done", False):
            return
        self._dim = dim
        self._init_done = True

    @property
    def dim(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        return f"P^{self._dim}"


class ProjectivePoint(Element):
    """
    A point in P^n represented by a nonzero homogeneous vector h ∈ R^{n+1},
    defined up to nonzero scalar multiples.
    """       
    def __init__(self, hvec: np.ndarray):
        h = np.asarray(hvec, dtype=float)
        if h.ndim != 1:
            raise ValueError("homogeneous vector must be 1D")
        if h.size == 0 or np.allclose(h, 0):
            raise ValueError("homogeneous vector must be nonzero")
        self._h = h
        super().__init__(ProjectiveSpace(h.size - 1))  # internal tag for compatibility

    # Accessors
    @property
    def h(self) -> np.ndarray:
        """Stored homogeneous representative (n+1,)."""
        return self._h
    
    def h_unit(self) -> np.ndarray:
        """Representative scaled to ||h|| == 1."""
        nrm = np.linalg.norm(self._h)
        if nrm == 0:
            raise ValueError("cannot normalize zero vector")
        return self._h / nrm
    
    def h_coord(self, i: int = -1, *, atol: float = 1e-12) -> np.ndarray:
        """
        Return a homogeneous rep scaled so h[i] == 1 (default: last coord).
        Raises if h[i] ≈ 0.
        """
        idx = i if i >= 0 else (self._h.size - 1)
        den = self._h[idx]
        if np.isclose(den, 0.0, atol=atol):
            raise ValueError(f"h[{idx}] ≈ 0; not in this chart")
        return self._h / den
       
    @property
    def dim(self) -> int:
        return self.space.dim
    
    def equals_up_to_scale(self, other: "ProjectivePoint", *, tol: float = 1e-12) -> bool:
        if not isinstance(other, ProjectivePoint) or other.space is not self.space:
            return False
        a, b = self.h, other.h
        denom = float(np.dot(b, b))  
        if denom == 0.0:
            return False
        s = float(np.dot(a, b))/ denom        # scale that best matches a ≈ s b
        return np.allclose(a, s * b, atol=tol, rtol=0.0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProjectivePoint) and self.equals_up_to_scale(other)
    
    def __repr__(self) -> str:
        return f"ProjectivePoint(h={self._h!r}, space=P^{self.space.dim})"
    
    
class ProjectiveTransformation(Transformation):
    """
    Projective linear map P^n -> P^m induced by an (m+1) x (n+1) matrix A,
    acting on homogeneous reps via [x] ↦ [A x]. Requires full column rank.
    """
    def __init__(self, A: np.ndarray):
        if not isinstance(A, np.ndarray) or A.ndim != 2:
            raise ValueError("A must be a 2D numpy array")
        m1, n1 = A.shape  # (m+1, n+1)
        if m1 <= 0 or n1 <= 0:
            raise ValueError("A must have shape (m+1, n+1) with m,n >= 0")
        if np.linalg.matrix_rank(A) != n1:
            raise ValueError("A must have full column rank (rank = n+1)")
        
        self._A = A
        self._m = m1 - 1  # codomain: P^m
        self._n = n1 - 1  # domain:   P^n
        super().__init__(ProjectiveSpace(self._n), ProjectiveSpace(self._m))
        
    @property
    def mat(self) -> np.ndarray:
        return self._A

    @property
    def dim(self) -> Tuple[int, int]:
        """(m, n) meaning P^n -> P^m."""
        return (self._m, self._n)
        
    def __call__(self, p: ProjectivePoint) -> ProjectivePoint:
        self.check_compatible(p)
        x = p.h
        y = self._A @ x
        if np.allclose(y, 0):
            raise ValueError("A @ p.h is (numerically) zero; map undefined at this point")
        return ProjectivePoint(y) 
    
    def __str__(self):
        return str(self._A)
    
    def __repr__(self) -> str:
        return f"ProjMap(P^{self._n} -> P^{self._m})"
    
    
class DualProjectiveSpace(Space):
    """Dual projective space (hyperplanes) of P^n (over R in this project)."""

    _cache: dict[int, "DualProjectiveSpace"] = {}

    def __new__(cls, dim: int):
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("dim must be a non-negative int")
        inst = cls._cache.get(dim)
        if inst is not None:
            return inst
        inst = super().__new__(cls)
        cls._cache[dim] = inst
        return inst

    def __init__(self, dim: int):
        # idempotent init (since __init__ runs even when returning a cached instance)
        if getattr(self, "_init_done", False):
            return
        self._dim = dim
        self._init_done = True

    @property
    def dim(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        return f"(P^{self._dim})*"
    
    
class ProjectiveHyperplane(Element):
    """
    A hyperplane in P^n given by a nonzero covector a ∈ (R^{n+1})*,
    defined up to nonzero scalar multiples.
    Incidence: point [h] lies on [a] iff a^T h = 0.
    """
    def __init__(self, avec: np.ndarray):
        a = np.asarray(avec, dtype=float)
        if a.ndim != 1:
            raise ValueError("covector must be 1D")
        if a.size == 0 or np.allclose(a, 0):
            raise ValueError("covector must be nonzero")
        self._a = a
        # dual space of P^n, n = a.size - 1
        super().__init__(DualProjectiveSpace(a.size - 1))

    # Accessors
    @property
    def a(self) -> np.ndarray:
        """Stored covector representative (n+1,)."""
        return self._a

    def a_unit(self) -> np.ndarray:
        """Representative scaled to ||a|| == 1."""
        nrm = np.linalg.norm(self._a)
        if nrm == 0:
            raise ValueError("cannot normalize zero covector")
        return self._a / nrm

    def a_coord(self, i: int = -1, *, atol: float = 1e-12) -> np.ndarray:
        """
        Return a covector rep scaled so a[i] == 1 (default: last coord).
        Raises if a[i] ≈ 0.
        """
        idx = i if i >= 0 else (self._a.size - 1)
        den = self._a[idx]
        if np.isclose(den, 0.0, atol=atol):
            raise ValueError(f"a[{idx}] ≈ 0; cannot use this coordinate for normalization")
        return self._a / den

    @property
    def dim(self) -> int:
        # hyperplanes live in the dual of P^n
        return self.space.dim

    def equals_up_to_scale(self, other: "ProjectiveHyperplane", *, tol: float = 1e-12) -> bool:
        if not isinstance(other, ProjectiveHyperplane) or other.space is not self.space:
            return False
        a, b = self.a, other.a
        denom = float(np.dot(b, b))
        if denom == 0.0:
            return False
        s = float(np.dot(a, b)) / denom  # scale that best matches a ≈ s b
        return np.allclose(a, s * b, atol=tol, rtol=0.0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProjectiveHyperplane) and self.equals_up_to_scale(other)

    def __repr__(self) -> str:
        return f"ProjectiveHyperplane(a={self._a!r}, space=(P^{self.space.dim})*)"


# ----- Operations ----- #
def incidence(p: ProjectivePoint, H: ProjectiveHyperplane, *, tol: float = 1e-12) -> bool:
    """
    Return True iff the projective point p lies on the hyperplane H,
    i.e. a^T h ≈ 0.

    p: ProjectivePoint in P^n
    H: ProjectiveHyperplane in (P^n)*
    """
    if not isinstance(p, ProjectivePoint):
        raise TypeError("p must be a ProjectivePoint")
    if not isinstance(H, ProjectiveHyperplane):
        raise TypeError("H must be a ProjectiveHyperplane")

    if p.dim != H.dim:
        # They live in different ambient dimensions, so incidence makes no sense
        return False

    a = H.a   # (n+1,)
    h = p.h   # (n+1,)

    val = float(np.dot(a, h))
    return np.isclose(val, 0.0, atol=tol, rtol=0.0)


def meet_unique(hyperplanes: Iterable[ProjectiveHyperplane], *, tol: float = 1e-12) -> ProjectivePoint:
    """
    Intersection point of hyperplanes in general position in P^n.

    We stack the covectors aᵢ as rows of a matrix A and compute its nullspace.
    If the nullspace is 1-dimensional (up to scale), we return the corresponding
    ProjectivePoint. Otherwise we raise.
    """
    hypers = list(hyperplanes)
    if not hypers:
        raise ValueError("meet_unique requires at least one hyperplane")

    dim0 = hypers[0].dim
    for H in hypers[1:]:
        if H.dim != dim0:
            raise ValueError("all hyperplanes must have the same dimension")

    A = np.stack([H.a for H in hypers], axis=0)  # shape (k, n+1)

    # SciPy nullspace: returns (n+1, r) where r is nullity
    N = null_space(A, rcond=tol)
    if N.size == 0 or N.shape[1] != 1:
        raise RuntimeError(
            f"Expected a unique intersection (nullity = 1), got nullity = {N.shape[1] if N.size else 0}"
        )

    x = N[:, 0]  # (n+1,)
    return ProjectivePoint(x)


def join_hyperplane(points: Iterable[ProjectivePoint], *, tol: float = 1e-12) -> ProjectiveHyperplane:
    """
    Hyperplane through a set of points in general position in P^n.

    We stack the homogeneous reps hᵢ as rows of a matrix P and compute the
    nullspace of P. A covector a in that nullspace satisfies P a = 0, i.e.
    aᵀ hᵢ = 0 for all i, so [a] is the hyperplane containing all the points.
    If the nullspace is 1-dimensional (up to scale), we return that hyperplane.
    Otherwise we raise.
    """
    pts = list(points)
    if not pts:
        raise ValueError("join_hyperplane requires at least one point")

    dim0 = pts[0].dim
    for p in pts[1:]:
        if p.dim != dim0:
            raise ValueError("all points must lie in the same projective space")

    P = np.stack([p.h for p in pts], axis=0)  # shape (k, n+1)

    N = null_space(P, rcond=tol)
    if N.size == 0 or N.shape[1] != 1:
        raise RuntimeError(
            f"Expected a unique hyperplane (nullity = 1), got nullity = {N.shape[1] if N.size else 0}"
        )

    a = N[:, 0]  # (n+1,)
    return ProjectiveHyperplane(a)



def dehomogenize(
    p: ProjectivePoint,
    infinity_plane: "ProjectiveHyperplane | None" = None,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Dehomogenize a projective point with respect to a chosen hyperplane at infinity.

    Given:
        - p: a ProjectivePoint in P^n
        - infinity_plane: a ProjectiveHyperplane H = [a], default is the canonical
          infinity plane x_n = 0 (covector e_n)

    Returns:
        A homogeneous vector x' with the same direction as p.h such that
        a^T x' = 1, where a is the covector of the infinity plane.

    Raises:
        TypeError / ValueError if types or dimensions mismatch, or if the point
        lies (numerically) on the infinity plane (a^T h ≈ 0).
    """
    if not isinstance(p, ProjectivePoint):
        raise TypeError("p must be a ProjectivePoint")

    n = p.dim  # P^n

    if infinity_plane is None:
        # canonical infinity hyperplane: x_n = 0  → a = e_n
        a = np.zeros(n + 1, dtype=float)
        a[-1] = 1.0
    else:
        if not isinstance(infinity_plane, ProjectiveHyperplane):
            raise TypeError("infinity_plane must be a ProjectiveHyperplane or None")
        if infinity_plane.dim != n:
            raise ValueError("point and infinity_plane must have the same dimension")
        a = infinity_plane.a

    h = p.h
    denom = float(np.dot(a, h))
    if np.isclose(denom, 0.0, atol=tol, rtol=0.0):
        raise ValueError("point lies (numerically) on the infinity plane; cannot dehomogenize")

    x_prime = h / denom
    
    assert abs(float(np.dot(a, x_prime)) - 1.0) <= 10 * tol
    return x_prime