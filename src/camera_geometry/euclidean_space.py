from typing import Iterable
from .space import Space, Transformation, Element, ElementPointcloud
import numpy as np


class EuclideanSpace(Space):
    _cache: dict[int, "EuclideanSpace"] = {}
    
    def __new__(cls, dim: int):
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("dim must be a non-negative int")
        # fast path
        inst = cls._cache.get(dim)
        if inst is not None:
            return inst
        obj = super().__new__(cls)
        cls._cache[dim] = obj
        return obj
    
    def __init__(self, dim: int):
        # make init idempotent (since it runs on every call)
        if getattr(self, "_init_done", False):
            return
        
        self._dim = dim
        self._init_done = True
        
    @property
    def dim(self) -> int:
        return self._dim

    def __repr__(self) -> str:
        return f"R^{self._dim}"
        
        
class EuclideanPoint(Element):
    def __init__(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError("matrix must be a numpy array")
        if x.ndim != 1:
            raise ValueError(f"The vector should only have one dimension")
        
        self._x = x
        super().__init__(EuclideanSpace(self._x.size))
        
    @property
    def x(self):
        return self._x
    
    @property
    def dim(self):
        return self._x.size
    
    def __sub__(self, other) -> np.ndarray:
        """Subtract two Euclidean points, returning the vector difference as a numpy array.
        """
        if not isinstance(other, EuclideanPoint):
            raise TypeError("Subtraction is only defined between EuclideanPoint instances")
        if self.dim != other.dim:
            raise ValueError("Points must have the same dimension for subtraction")
        return self.x - other.x
    
    def __add__(self, displacement: np.ndarray):
        """Add a displacement vector (numpy array) to this Euclidean point, returning a new EuclideanPoint.
        """
        if not isinstance(displacement, np.ndarray):
            raise TypeError("Displacement must be a numpy array")
        if displacement.ndim != 1 or displacement.size != self.dim:
            raise ValueError("Displacement must be a 1D numpy array of the same dimension as the point")
        return EuclideanPoint(self.x + displacement)
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply by a scalar (int or float)")
        return EuclideanPoint(self.x * scalar)
    
    def __eq__(self, other):
        if isinstance(other, EuclideanPoint):
            return np.array_equal(self.x, other.x)
        return False
    
    def __repr__(self) -> str:
        return f"EuclideanPoint(x={self._x!r}, space=R^{self.space.dim})"
    
    
class EuclideanTransformation(Transformation):
    def __init__(self, matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("matrix must be a numpy array")
        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2D (m x n)")       
        self._dim_out, self._dim_in = matrix.shape[0], matrix.shape[1]
        
        super().__init__(EuclideanSpace(self._dim_in), EuclideanSpace(self._dim_out))
        self._matrix = matrix
        
    @property
    def dim(self) -> tuple[int, int]:
        """(m, n) meaning R^n -> R^m."""
        return (self._dim_out, self._dim_in)
    
    @property
    def mat(self):
        return self._matrix
        
    def __call__(self, p: EuclideanPoint):
        self.check_compatible(p)
        y = self._matrix @ p.x
        return EuclideanPoint(y)  
    
    def __str__(self):
        return str(self._matrix)
    
    def __repr__(self):
        return f"EuclideanTransformation(matrix={self._matrix!r})"
    

class EuclideanPointcloud(ElementPointcloud):
    def __init__(self, points: Iterable[EuclideanPoint]=None, space: Space=None, check_points=True):
        if points is not None and len(points) == 0:
            points = None
        super().__init__(points=points, space=space, check_points=check_points)
        self._dim = self.space.dim

    def transform(self, transformation: Transformation):
        dim_out, dim_in = transformation.dim
        if dim_in != self._dim:
            raise ValueError(f"Transformation's dimensions ({dim_in}) are incompatible with that of the pointcloud {self._dim}")
        super().transform(transformation)
        self._dim = dim_out

    def add(self, point: EuclideanPoint):
        if not isinstance(point, EuclideanPoint):
            raise ValueError("point is expected to be of type EuclideanPoint")
        if point.space != self.space:
            raise ValueError(f"point is expected to belong to the same space as the pointcloud")
        super().add(point)

    def pop(self, i: int):
        if i >= len(self._points):
            raise ValueError(f"index i is required to be smaller than the length of the pointcloud {len(self._points)}")
        return super().pop(i)
    
    @property
    def mat(self) -> np.ndarray:
        """
        Euclidean Coordinates of all points as an array of shape (n, N),
        where each column is the Euclidean vector of a point.
        """
        if not self._points:
            n1 = self.space.dim
            return np.zeros((n1, 0), dtype=float)
        return np.stack([p.x for p in self._points], axis=1) # points stored in cols

    @property
    def points(self):
        return self._points
    
    @property
    def dim(self) -> int:
        return self.space.dim
    
    @staticmethod
    def from_ply(file: str):
        """
        Returns a pointcloud from a ply pointcloud file.

        Currently assumes:
          - ASCII PLY
          - a single 'vertex' element with at least x, y, z as the first 3 properties
        """
        points: list[EuclideanPoint] = []
        vertex_count: int | None = None

        with open(file, "r") as fp:
            # Read header
            line = fp.readline()
            if not line.startswith("ply"):
                raise ValueError("Not a PLY file (missing 'ply' magic)")

            while True:
                line = fp.readline()
                if not line:
                    raise ValueError("Unexpected EOF while reading PLY header")

                line_stripped = line.strip()

                # Capture vertex count from element line
                # e.g. "element vertex 100000"
                if line_stripped.startswith("element vertex"):
                    parts = line_stripped.split()
                    if len(parts) != 3:
                        raise ValueError(f"Malformed 'element vertex' line: {line_stripped!r}")
                    try:
                        vertex_count = int(parts[2])
                    except ValueError as e:
                        raise ValueError(f"Invalid vertex count in PLY header: {line_stripped!r}") from e

                if line_stripped == "end_header":
                    break

            # Now read vertex_count lines (if we know it), or until EOF
            if vertex_count is None:
                # Fallback: read until EOF
                for line in fp:
                    splitted = line.split()
                    if not splitted:
                        continue
                    vec = np.array(splitted[0:3]).astype(float)
                    point = EuclideanPoint(vec)
                    points.append(point)
            else:
                for _ in range(vertex_count):
                    line = fp.readline()
                    if not line:
                        break
                    splitted = line.split()
                    if not splitted:
                        continue
                    vec = np.array(splitted[0:3]).astype(float)
                    point = EuclideanPoint(vec)
                    points.append(point)
                    
        return EuclideanPointcloud(points=points)
               
    def __len__(self):
        return len(self._points)
    
    def __iter__(self):
        return iter(self._points)
    
    def __getitem__(self, i):
        return self._points[i]
    
    def __repr__(self) -> str:
        return f"EuclideanPointcloud(size={len(self._points)}, space=E^{self.space.dim})"