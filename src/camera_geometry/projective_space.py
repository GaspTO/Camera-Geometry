from __future__ import annotations
import numpy as np
from scipy.linalg import null_space
from typing import List, Union


def check_vector(vec: np.ndarray, name: str):
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"vec should be of type np.ndarray but is of type {type(vec)}")
    if len(vec.shape) != 1:
        raise ValueError(f"The vector should only have one dimension")
    if np.count_nonzero(vec) == 0:
        raise ValueError(f"{name} can not be 0")
    

class Point:
    def __init__(self, vec: np.ndarray):
        check_vector(vec, "vec")
        self._vec = vec/np.linalg.norm(vec) # Store unit vector
        self._dim = vec.shape[0] - 1
        
    def vec(self):
        assert abs(np.linalg.norm(self._vec) - 1) < 0.001
        return self._vec
       
    def dim(self):
        return self._dim
    
    def contained(self, hyperplane: Hyperplane):
        if not isinstance(hyperplane, Hyperplane):
            raise ValueError("point needs to be of type Point")
        if hyperplane.dim() != self.dim():
            raise ValueError(f"point needs to be of dim {self.dim()}")
    
    @staticmethod
    def find_point(planes: List[Hyperplane]):
        Pi = np.array([pi.vec() for pi in planes]) # each point is a row
        N = null_space(Pi)
        if N.shape[1] != 1:
            raise RuntimeError("Expected a unique intersection (nullity ≠ 1).")
        assert abs(1 - np.linalg.norm(N)) < 0.0001
        x = N[:, 0]
        return Point(x)
       
    def __eq__(self, other):
        if isinstance(other,Point):
            return np.array_equal(self.vec(),other.vec())
        return False
    
    def __str__(self):
        return str(self._vec)
    
    
class Hyperplane:
    def __init__(self, vec:np.ndarray):
        check_vector(vec, "vec")
        self._vec = vec/np.linalg.norm(vec) # Store unit vector
        self._dim = vec.shape[0] - 1
        
    def vec(self):
        assert abs(np.linalg.norm(self._vec) - 1) < 0.001
        return self._vec
    
    def dim(self):
        return self._dim
    
    def contains(self, point: Point):
        if not isinstance(point, Point):
            raise ValueError("point needs to be of type Point")
        if point.dim() != self.dim():
            raise ValueError(f"point needs to be of dim {self.dim()}")
        
        return abs(np.dot(self.vec(),point.vec())) < 1e-6
    
    @staticmethod
    def find_plane(points: list["Point"]) -> "Hyperplane":
        X = np.array([p.vec() for p in points])           # rows = points
        N = null_space(X)                                 # π satisfies X @ π = 0
        if N.shape[1] != 1:
            raise RuntimeError("Expected a unique plane (nullity ≠ 1).")
        π = N[:, 0]
        return Hyperplane(π)
                
    def __eq__(self, other):
        if isinstance(other,Hyperplane):
            return np.array_equal(self.vec(),other.vec())
        return False
    
    def __str__(self):
        return str(self._vec)
        
    
class ProjectiveSpace:
    def __init__(self, dim: int=None, infty_hyperplane: Hyperplane=None):
        if dim is not None and infty_hyperplane is not None:
            raise ValueError("Do not specify the dimension if the infinity hyperplane is passed as argument")  
        if dim is None and infty_hyperplane is None:
            raise ValueError("Either specify the dimension of the projective space of the hyperplane at infinity")
    
        if dim:
            arr = np.zeros(dim+1)
            arr[-1] = 1
            self._infty_hyperplane = Hyperplane(arr)    # canonical hyperplane (0,0,...,1)
            self._dim = dim
        else:
            self._infty_hyperplane = infty_hyperplane
            self._dim = infty_hyperplane.dim() 
          
    def dim(self):
        return self._dim
    
    def dehomogenize(self, point: Point):
        if point.dim() != self.dim():
            raise ValueError("Point is of a different dimension than the projective space")
        x = point.vec()
        pi = self._infty_hyperplane.vec()
        dehomogenized_x = x / np.dot(pi,x)
        assert abs(np.dot(dehomogenized_x,pi) - 1.0) < 0.001
        return dehomogenized_x
         
    def is_ideal_point(self, point: Point):
        if not isinstance(point, Point):
            raise ValueError("point needs to be of type Point")
        if point.dim() != self.dim():
            raise ValueError(f"point needs to be of dim {self.dim()}")
        return self._infty_hyperplane.contains(point)
    
class Transformation:
    def __init__(self, matrix: np.ndarray):
        if len(matrix.shape) != 2:
            raise ValueError("Matrix shape is not (x,y)")
        self._dim_out, self._dim_in = matrix.shape[0]-1, matrix.shape[1]-1
        if not np.isclose(matrix[-1][-1],0.0):
            self._matrix = matrix / matrix[-1][-1] # normalize such that the last element is a 1
        
    def dim(self):
        return self._dim_out, self._dim_in
    
    def mat(self):
        return self._matrix 
        
    def __call__(self, x: Union[Point, Hyperplane]):
        """ It outputs in the same shape as the input"""
        if not isinstance(x, Point):
            raise ValueError("Point is expected")
                
        if x.dim() != self._dim_in:
            raise ValueError("Point dimension is incompatible with matrix")
        y = self._matrix @ x.vec()
        return Point(y)  

    def is_singular(self):
        return np.linalg.matrix_rank(self._matrix).item() != self._matrix.shape[0]
    
    def __str__(self):
        return str(self._matrix)
    
    
class Composition(Transformation):
    def __init__(self, t1: Transformation, t2: Transformation):
        if t1.dim()[1] != t2.dim()[0]:
            raise ValueError("The transformations matrix have incompatible dimensions")
        t3_mat = t1.mat() @ t2.mat()
        super().__init__(t3_mat)
        
        
class Homography(Transformation):
    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix=matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("An homography matrix is expected to be square")
        if np.linalg.matrix_rank(matrix.astype(float)) != matrix.shape[0]:
            raise ValueError("An homography is expected to be non singular")
    
    def get_fixed_points(self):
        _, V = np.linalg.eig(self._matrix)
        points = []
        for i in range(V.shape[1]):
            points.append(Point(V[:,i]))
        return points
    
    def is_affinity(self):
        return np.all(self._matrix[-1, :-1] == 0) and (self._matrix[-1, -1] != 0) and self._dim_in == self._dim_out
   
    def get_inverse(self):
        return Homography(np.linalg.inv(self._matrix))


if __name__ == "__main__":
    ## Points
    p1 = Point(np.array([1,2,3,1]))
    p2 = Point(np.array([2,4,6,2]))
    p3 = Point(np.array([1,2,2,1]))
    p4 = np.array([1,2,3,1])
    assert p1 == p2
    assert not (p2 == p3)
    assert not (p3 == p4)
    print("Points success")
    
   
    ## Hyperplane
    p1 = Point(np.array([1,2,3,1]))
    p2 = Point(np.array([2,4,6,1]))
    p3 = Point(np.array([1,2,2.,1]))
    pi = Hyperplane.find_plane([p1,p2,p3])
    
    p3 = Point(np.array([2,4,6.,1]))
    try:
        pi = Hyperplane.find_plane([p1,p2,p3])
    except Exception as e:
        print("ok")
    
    ## Projective Space
    X = Point(np.array([1,2,3,1]))
    
    plane = Hyperplane(np.array([1,2,3,1]))    
    space = ProjectiveSpace(infty_hyperplane=plane)
    deh_x = space.dehomogenize(X)
    assert abs(np.dot(deh_x,plane.vec()) - 1.0) < 0.001
    print("Projective space success")
    
    plane = Hyperplane(np.array([0,0,0,1]))
    canonical_space = ProjectiveSpace(infty_hyperplane=plane)
    deh_x = canonical_space.dehomogenize(X)
    assert abs(np.dot(deh_x,plane.vec()) - 1.0) < 0.001
    assert canonical_space.is_ideal_point(Point(np.array([2,3,2,0])))
    
    
    ## Transformations
    t1 = Transformation(
        np.array([
            [2., 0., 0., 0.],
            [0., 3., 0., 0.],
            [0., 0., 4., 0.],
            [1., 2., 3., 1.]
        ])
    )
    assert t1.dim() == (4,4)
    
    t2 = Transformation(
        np.array([
            [1., 0., 0.],
            [4., 3., 0.],
            [5., 0., 0.],
            [1., 2., 1.]
        ])
    )
    assert t2.dim() == (4,3)
    
    t3 = Composition(t1,t2)
    assert t3.dim() == (4,3)
    assert abs(t3.mat() - np.array([
                        [2., 0., 0.],
                        [12., 9., 0.],
                        [20., 0., 0.],
                        [25., 8., 1.]
                    ])).sum() < 0.001
    homography = Homography(t1.mat())
    for e in homography.get_fixed_points():
        print(e)