from camera_geometry.projective_space import ProjectivePoint, ProjectiveTransformation, ProjectiveSpace, ProjectivePointcloud
from camera_geometry.euclidean_space import EuclideanPoint, EuclideanTransformation
from camera_geometry.space import Composition
import numpy as np
from itertools import compress

class PinholeCamera:
    """
    A pinhole camera model with intrinsic and extrinsic parameters.
    
    Coordinate system conventions:
 
    3D Camera coordinate system (right-handed):
    
            Z (forward, viewing direction)
           /
          /
         o-----> X (right)
         |
         |
         Y (down)
      
    2D Image coordinate system:
         o-----> u (column, right)
         |
         |
         v
         v (row, down)
         
    Using v downwards for image rows is common in image processing, since
    it matches the way images are typically stored in memory. Thus,
    the pixel at the top-left corner of the image has coordinates (u=0, v=0).
    """
    def __init__(self, fx, fy, cx, cy, height=None, width=None, debug=False, optimize=True):
        self._debug = debug
        self._height = height
        self._width = width
        
        self._intrinsics = ProjectiveTransformation(
            np.array([
                [fx, 0 , cx],
                [0,  fy, cy],
                [0, 0, 1]
            ]))
        
        ## Prespective Projection matrix
        self._projection = ProjectiveTransformation(np.array([ 
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]))
           
        ## Default extrinsics [R(0º)|-C=0]. Rigid transformation
        # in other words, the camera points at the z-axis in the world and is
        # at the origin.
        self._extrinsics = ProjectiveTransformation( 
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0 ,0 ,0, 1]
            ]))
                
        self._optimize = optimize
        self.update_camera_matrix()
  
    def __call__(self,
                 pointcloud: ProjectivePointcloud,
                 height=None,
                 width=None,
                 background=None,
                 foreground=None,
                 batch_size= 1) -> np.array:
        """ Projects a 3D pointcloud into a 2D image plane.
        Args:
            pointcloud: ProjectivePointcloud in P^3 (3D points)
            height: image height in pixels (overrides camera default if given)
            width: image width in pixels (overrides camera default if given)
        Returns:
            2D image as a numpy array of shape (height, width).
        """
        if height is None:
            height = self._height
        if width is None:
            width = self._width
        if height is None or width is None:
            raise ValueError("Height and width must be specified either at camera creation or call time.")

        batch_num = len(pointcloud) // batch_size + (len(pointcloud) % batch_size > 0)
        pointcloud_mat = pointcloud.mat  # shape (4, N)

        image = np.full((height,width), dtype=object, fill_value=background)
        z_values = np.full((height,width), np.inf)
        for b in range(batch_num):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(pointcloud))
            batch_points = pointcloud_mat[:, start_idx:end_idx]  # shape (4, batch_size)
            X, Y, Depth = self._camera.mat @ batch_points
            X, Y = X / Depth, Y / Depth  # Normalize by depth
            X, Y = np.round(X).astype(int), np.round(Y).astype(int)  # Pixel coordinates         
            U, V = self.from_xy_to_uv(X, Y, height, width) 
            
            # conditions
            in_bounds = (U >= 0) & (U < width) & (V >= 0) & (V < height)
            valid_depth = Depth > 0 # Convert to (u,v) coordinates
            idxs = in_bounds & valid_depth
            closer = np.zeros_like(idxs, dtype=bool)
            closer[idxs] = z_values[V[idxs], U[idxs]] > Depth[idxs] # Points closer to the camera than current z-buffer
            mask = in_bounds & valid_depth & closer
            
            z_values[V[mask], U[mask]] = Depth[mask]  # Update z-buffer
            if foreground is not None:
                image[V[mask], U[mask]] = foreground
            else:
                current_slice = pointcloud.points[start_idx:end_idx]
                valid_points = list(compress(current_slice, mask))
                image[V[mask], U[mask]] = valid_points
        return image     

    
    def update_camera_matrix(self) -> None:
        if self._optimize:
            self._camera = ProjectiveTransformation(self._intrinsics.mat @ self._projection.mat @ self._extrinsics.mat)  
        else:
            self._camera = Composition(self._intrinsics, Composition(self._projection, self._extrinsics))    

    def get_rotation_translation(self) -> tuple[EuclideanTransformation, EuclideanPoint]:
        """ Returns the rotation matrix and camera position in world coordinates.
        """
        R = self._extrinsics.mat[0:3, 0:3]
        r_t = self._extrinsics.mat[0:3, 3] # This is the position of the world origin in camera coords
        C_world = R.T @ -r_t # Camera position in world coords
        return EuclideanTransformation(R), EuclideanPoint(C_world)
    
    get_orientation_position = get_rotation_translation  # Alias
    
    def get_forward_up_right(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R, _ = self.get_rotation_translation()
        R = R.mat
        right   = R[0, :]
        up      = R[1, :]
        forward = R[2, :]
        return forward, up, right
    
    def set_position(self, C_world: EuclideanPoint) -> None:
        """
        Set the camera center in world coordinates.
        Args:
            C_world (EuclideanPoint): New camera position in world coordinates.
        """
        R, _ = self.get_rotation_translation()
        self._extrinsics = self.assemble_extrinsics(R.mat, C_world.x)
        self.update_camera_matrix()
        
    def set_orientation(self, R: EuclideanTransformation) -> None:
        """ Sets the camera orientation given a rotation matrix R.
        Note: The camera position remains unchanged.
        Args:
            R (EuclideanTransformation): Rotation matrix representing new camera orientation. 
                The matrix should be 3x3, and should correspond to a transformation from World to Camera coordinates.
                An easy way to come up with this matrix is to encode the axis directions of the camera as rows:
        """
        _, cam_pos = self.get_rotation_translation()
        self._extrinsics = self.assemble_extrinsics(R.mat, cam_pos.x)
        self.update_camera_matrix()
        
    def look_at(self, target: EuclideanPoint, up: np.ndarray =None) -> None:
        """ Orient the camera to look at a given target point in world coordinates.
        Args:
            target (EuclideanPoint): The point in world coordinates to look at.
            up (np.ndarray, optional): An optional up vector to define the camera's vertical orientation.
                If not provided, a default up vector of [0, -1, 0] is used.
        """
        if not isinstance(target, EuclideanPoint):
            raise ValueError("Target must be a EuclideanPoint.")
        if up is None:
            up_vector = np.array([0.0, -1.0, 0.0], dtype=float) # The y axis points downwards by default. Calling it up is a misdomer.
        elif not isinstance(up, np.ndarray):
            raise ValueError("Up vector must be a numpy array or None.")
        else:
            up_vector = np.asarray(up, dtype=float)
    
        # Camera center in world coordinates (EuclideanPoint)
        _, C = self.get_rotation_translation()

        # Forward / view direction: from camera to target
        dir_vector = target - C           # uses EuclideanPoint.__sub__, returns np.ndarray
        dir_norm = np.linalg.norm(dir_vector)
        if dir_norm == 0:
            raise ValueError("look_at: target coincides with camera position.")
        dir_vector = dir_vector / dir_norm

        # Right vector: up × forward
        right_vector = np.cross(up_vector, dir_vector)
        right_norm = np.linalg.norm(right_vector)
        if right_norm == 0:
            raise ValueError("look_at: 'up' vector is parallel to view direction.")
        right_vector = right_vector / right_norm

        # Recompute up to ensure orthogonality: forward × right
        up_vector = np.cross(dir_vector, right_vector)

        # Build rotation and apply
        R_new = self.assemble_rotation(
            forward=dir_vector,
            up=up_vector,
            right=right_vector,
        )
        self.set_orientation(EuclideanTransformation(R_new))
    
    @staticmethod
    def from_xy_to_uv(x, y, height, width):
        """Convert from plane coordinates (x, y) to array indices (u, v).

        Here we adopt the image-processing convention:
        - x: column index, increases to the right
        - y: row index, increases downwards
        - u, v follow the same convention, so (u, v) = (x, y).
        """
        return x, y
    
    @staticmethod
    def from_uv_to_xy(u, v, height, width):
        """Convert from array indices (u, v) to plane coordinates (x, y).

        Here we adopt the image-processing convention:
        - x: column index, increases to the right
        - y: row index, increases downwards
        - u, v follow the same convention, so (u, v) = (x, y).
        """
        return u, v
    
    @staticmethod
    def assemble_rotation(forward: np.ndarray, up: np.ndarray, right: np.ndarray):
        R = np.array([
            [right[0], right[1], right[2]],
            [up[0],    up[1],    up[2]],
            [forward[0],   forward[1],   forward[2]]
        ])
        return R
     
    @staticmethod
    def assemble_extrinsics(R: np.ndarray, cam_pos: np.ndarray):
        """ Assembles extrinsics matrix from rotation matrix and
         camera position in world coordinates."""
        translation = -R @ cam_pos
        extrinsics = ProjectiveTransformation(
            np.array([
                [R[0,0], R[0,1], R[0,2], translation[0]],
                [R[1,0], R[1,1], R[1,2], translation[1]],
                [R[2,0], R[2,1], R[2,2], translation[2]],
                [0,      0,      0,      1]
            ]))
        return extrinsics
    
    def ideal_array(self, height, width) -> np.ndarray:
        """Builds a height x width array of ideal points in the image plane representing the
        projection rays of each pixel.
        Args:
            height: image height in pixels
            width: image width in pixels
        Returns:
            A numpy array of shape (height, width) where each element is a ProjectivePoint
            representing the ideal point (direction) corresponding to that pixel."""
        M = self._camera.mat[:, 0:3]
        M_inv = np.linalg.inv(M)

        # Build grid of (u, v)
        us, vs = np.meshgrid(np.arange(width), np.arange(height))  # shape (H, W)

        # Convert to (x, y); here this is identity, but we keep the op explicit
        xs, ys = self.from_uv_to_xy(us, vs, height, width)  # same shapes

        # Stack homogeneous image coords: shape (3, H*W)
        img_coords = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()], axis=0)

        # Back-project all directions at once: shape (3, H*W)
        world_dirs = M_inv @ img_coords  # rows: Xw, Yw, Zw

        # Ideal points: shape (4, H*W)
        homo_dirs = np.vstack([world_dirs, np.zeros((1, world_dirs.shape[1]))]) # homogeneous directions
        image = np.empty((height, width), dtype=object)
        flat = [ProjectivePoint(homo_dirs[:, i]) for i in range(homo_dirs.shape[1])]
        image[:, :] = np.array(flat, dtype=object).reshape(height, width)
        
        return image