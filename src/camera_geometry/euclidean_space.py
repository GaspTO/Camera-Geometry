from .space import Space, Transformation, Element
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