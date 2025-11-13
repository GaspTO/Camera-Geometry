# Camera Geometry (Python)

Small, focused utilities for projective geometry in homogeneous coordinates: points, hyperplanes, projective spaces, linear transformations, and homographies — with helpers to find intersections via null spaces.

This library is for educational purposes only. It details part of my journey in learning these concepts.


## Installation
This project targets Python **3.9+** (recommended) with **NumPy** and **SciPy**.

### Using `uv` (recommended)
```bash
# inside the repo root
uv sync                   # installs deps and the package (editable by default)
```

### Using pip
```bash
pip install -e .          # editable install for local dev
```

## Quick start
```python
import numpy as np
from projective import Point, Hyperplane, ProjectiveSpace, Homography

# Points / planes in P^2
x1 = Point(np.array([1.0, 0.0, 1.0]))
x2 = Point(np.array([0.0, 1.0, 1.0]))
pi = Hyperplane.find_plane([x1, x2])   # plane through x1 and x2
assert pi.contains(x1) and pi.contains(x2)

# Intersection of planes
p1 = Hyperplane(np.array([1.0, 0.0, -1.0]))
p2 = Hyperplane(np.array([0.0, 1.0, -1.0]))
x = Point.find_point([p1, p2])
assert p1.contains(x) and p2.contains(x)

# Projective space (canonical π∞ = [0,0,1])
P = ProjectiveSpace(dim=2)
affine = P.dehomogenize(Point(np.array([2.0, -3.0, 1.0])))

# Homography (must be square and non‑singular)
H = Homography(np.array([
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]))
y = H(Point(np.array([1.0, 2.0, 1.0])))
```

---