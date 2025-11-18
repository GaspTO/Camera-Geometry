"""
This module shows how to implement the backwarper algorithm in order
to apply an homography to an image. 

The transformations are applied to the (u,v) axis, corresponding to pixel locations

Image (u, v) axes for array img[v, u]:

        u →
      0   1   2   ...
v ↓  ┌───┬───┬───┬───
  0  │   │   │   │
     ├───┼───┼───┼───
  1  │   │   │   │
     ├───┼───┼───┼───
  2  │   │   │   │
  .  │   │   │   │
  .  │   │   │   │

- u: horizontal axis (columns), increases to the right
- v: vertical axis (rows), increases downward
- Pixel access: img[v, u]

"""
from .projective_space import ProjectivePoint, is_ideal, dehomogenize, Homography
import numpy as np
from typing import Tuple


class ImageWarper:
    """
    Abstract calss which modifies images given an homography. 
    It uses the inverseWarp algorithm.
    """
    def __init__(self, warp_transform: Homography):
        if warp_transform.dim != (2,2):
            raise ValueError("Image transformer expects a 2-dimensional homography")
        self._warp_transform = warp_transform
            
    def __call__(self,
                 image: np.ndarray,
                 background=0,
                 new_dimension: Tuple = None,
                 ideal_point_value = None) -> np.ndarray:
        """Applies the transformation to an image.

        Args:
            image (np.ndarray): image array
            background (int, optional): Defaults to 0.
            new_dimension (Tuple, optional): Dimension of warped image.
                Defaults to the same as input's.
            ideal_point_value (_type_, optional): Pixel value for the ideal line.
                Defaults to not drawing the line.

        Returns:
            np.ndarray : warped image
        """
        
        if new_dimension is None:
            new_img = np.zeros_like(image)
        else:
            new_img = np.full_like(new_dimension,background)
        homography_inv = self._warp_transform.inverse()
    
        # Iterate over the image
        for target_v in range(new_img.shape[0]):
            for target_u in range(new_img.shape[1]):
                target_point = ProjectivePoint(np.array([target_u, target_v,1]))
                source_point = homography_inv(target_point)
                if not is_ideal(source_point): 
                    new_img[target_v][target_u] = self._calculate_new_value(image, source_point, background)
                elif ideal_point_value is not None:
                    new_img[target_v][target_u] = ideal_point_value
        return new_img
            
    def _calculate_new_value(self, img, source_point, background):
        raise NotImplementedError


class ImageWarperNN(ImageWarper):
    """Inverse image warper which uses the nearest neighbor interpolation.
    The domain of the transformation is the (u,v) axis of the array."""
    def __init__(self, warp_transform: Homography):
        super().__init__(warp_transform)
    
    def _calculate_new_value(self, img, source_point, background):
        u, v, _ = dehomogenize(source_point)
        u, v = int(np.round(u)), int(np.round(v)) # nearest neighbor
        if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
            return img[v][u]
        else:
            return background
    
        
class ImageWarperBilinear(ImageWarper):
    """Inverse image warper which uses bilinear interpolation.
    The domain of the transformation is the (u,v) axis of the array."""
    def __init__(self, warp_transform: Homography):
        super().__init__(warp_transform)
    
    def _calculate_new_value(self, img, source_point, background):
        u, v, _ = dehomogenize(source_point)
        u0, v0 = int(np.floor(u)), int(np.floor(v))
        du, dv = u-u0, v-v0
        H,W = img.shape[0], img.shape[1]
        if 0 <= u0 < W-1 and 0 <= v0 < H-1: # bilinear interpolation
            I00 = img[v0][u0].astype(np.float32)
            I10 = img[v0][u0+1].astype(np.float32)
            I01 = img[v0+1][u0].astype(np.float32)
            I11 = img[v0+1][u0+1].astype(np.float32)
            
            val = (1-du)*(1-dv)*I00 + du*(1-dv)*I10 + (1-du)*dv*I01 + du*dv*I11
            return val.astype(img.dtype)
        elif 0 <= u0 < W and 0 <= v0 < H:   # border: fall back to nearest
            return img[v0, u0]
        else:
            return background