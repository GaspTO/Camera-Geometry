from abc import ABC, abstractmethod

class Space(ABC):
    def contains(self, x) -> bool:
        # Default: accept Points tagged with this space
        return isinstance(x, Element) and x.space is self
   

class Element(ABC):
    def __init__(self, space: Space):
        if not isinstance(space, Space):
            raise ValueError("space is expected to be a Space")
        self.space = space
        
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