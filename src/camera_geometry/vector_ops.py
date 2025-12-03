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

def multiply_quaternions(q1, q2):
    # Assumes q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])