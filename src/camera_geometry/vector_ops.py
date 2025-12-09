import numpy as np

def vector_projection(v: np.ndarray,n: np.ndarray):
    """ Projects vector v onto vector n."""
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

## Quaternion utils
def multiply_quaternions(q1, q2):
    """ Hamilton product of two quaternions."""
    # Assumes q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def rotate_vector_by_quaternion(q, v):
    """
    Rotates a 3D vector v using a quaternion q.
   
    While, in theory, this can be done via quaternion multiplication (q * V * q^*),
    where V is the vector represented as a pure quaternion [0, v_x, v_y, v_z],
    it's often more efficient to use the following formula:
        v_rotated = v + 2 * cross(r, cross(r, v) + w * v)
    where q = [w, r_x, r_y, r_z]
    
    Args:
        q (np.array): [w, x, y, z] (Scalar First, Normalized)
        v (np.array): [x, y, z] (3D Vector)
    
    Returns:
        np.array: Rotated 3D vector
    """
    # 1. Extract parts
    w = q[0]
    r = q[1:]  # Vector part (x, y, z)

    t = 2 * np.cross(r, v)   
    v_rotated = v + 2 * np.cross(r, np.cross(r, v) + w * v)
    
    return v_rotated
