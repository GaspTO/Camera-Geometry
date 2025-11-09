import numpy as np

def vector_projection(v: np.ndarray,n: np.ndarray):
    normed_n = n/np.linalg.norm(n)
    proj_v = np.dot(v,normed_n) * normed_n 
    return proj_v
   
def colinear(a: np.ndarray, b: np.ndarray, tol=1e-6) -> bool:
    """True if a and b point in the same projective direction."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return False
    au = a / na
    bu = b / nb
    return np.allclose(au, bu, atol=tol) or np.allclose(au, -bu, atol=tol)
