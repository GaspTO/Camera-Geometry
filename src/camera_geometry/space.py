from abc import ABC, abstractmethod
from typing import Iterable

class Space(ABC):
    def contains(self, x) -> bool:
        # Default: accept Points tagged with this space
        return isinstance(x, Element) and x.space is self
   

class Element(ABC):
    def __init__(self, space: Space, content: object = None):
        if not isinstance(space, Space):
            raise ValueError("space is expected to be a Space")
        self.space = space
        self._content = content
        
    @property
    def content(self) -> object:
        return self._content
    
    def __repr__(self):
        return f"Element(space={id(self.space)})"
    
        
class Transformation(ABC):
    def __init__(self, domain_space: Space, codomain_space: Space):
        if not isinstance(domain_space, Space) or not isinstance(codomain_space, Space):
            raise ValueError("domain_space and codomain_space must be Space instances")
        self.domain_space = domain_space
        self.codomain_space = codomain_space
        
    def check_compatible(self, x):
        if not self.domain_space.contains(x):
            raise TypeError("Input is not compatible with transformation's domain")
        
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError
    
    
class Composition(Transformation):
    """t1 ∘ t2 (apply t2 first, then t1)."""
    def __init__(self, t1: Transformation, t2: Transformation):
        if t2.codomain_space != t1.domain_space:
            raise ValueError("The transformations are incompatible for composition t1 ∘ t2")
        self.t1 = t1
        self.t2 = t2
        super().__init__(domain_space=t2.domain_space, codomain_space=t1.codomain_space)
        
    def __call__(self, x):
        self.check_compatible(x)
        y = self.t2(x)
        z = self.t1(y)
        assert self.codomain_space.contains(z), "Output not in codomain"
        return z
        
    def __str__(self):
        return f"({str(self.t1)})\n∘\n({str(self.t2)})"
    
    def __repr__(self):
        return f"Composition({repr(self.t1)}, {repr(self.t2)})"
    
class ElementPointcloud(Element):
    """Generic pointcloud of Elements sharing the same Space."""
    def __init__(self, points: Iterable[Element] | None = None, space: Space = None, check_points: bool = True):
        pts = [] if points is None else list(points)
        if pts and space is not None:
            raise ValueError("Only pass space when initializing an empty pointcloud.")
        if pts:
            space = pts[0].space
            if check_points:
                for p in pts:
                    if p.space is not space:
                        raise ValueError("All points must belong to the same space.")
        else:
            if not isinstance(space, Space):
                raise TypeError("space must be provided for an empty pointcloud.")
        self._points = pts
        super().__init__(space=space)

    def transform(self, transformation: Transformation):
        for p in self._points:
            transformation.check_compatible(p)
        new_points = [transformation(p) for p in self._points]
        target_space = transformation.codomain_space
        for p in new_points:
            if p.space is not target_space:
                raise TypeError("Transformed point does not belong to the transformation codomain.")
        self._points = new_points
        self.space = target_space

    def add(self, point: Element):
        if not isinstance(point, Element):
            raise ValueError("point must be an Element.")
        if point.space is not self.space:
            raise ValueError("point must belong to the same space as the pointcloud.")
        self._points.append(point)

    def pop(self, i: int):
        if i >= len(self._points):
            raise ValueError("index out of range for pointcloud.")
        return self._points.pop(i)

    def __len__(self):
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __getitem__(self, i):
        return self._points[i]

    def __repr__(self) -> str:
        return f"ElementPointcloud(size={len(self._points)}, space={id(self.space)})"
