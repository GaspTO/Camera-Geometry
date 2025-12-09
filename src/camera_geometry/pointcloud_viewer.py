import matplotlib.pyplot as plt
import numpy as np

from camera_geometry.camera import PinholeCamera
from camera_geometry.euclidean_space import EuclideanPoint, EuclideanPointcloud
from camera_geometry.vector_ops import multiply_quaternions, rotate_vector_by_quaternion


class PointcloudViewer:
    """
    Example:
        viewer = PointcloudViewer("pointcloud.ply")
        viewer.show()
    
    Use keys to translate the camera:
        up, down, left, right arrows: move camera in the image plane
        mouse wheel: move camera forward/backward
    Mouse controls:
        Left click + drag: rotate camera around an anchor point (arcball rotation)
        Right click on a point: set anchor point for rotation
    """
    def __init__(
        self,
        ply_path: str,
        width: int = 1000,
        height: int = 1000,
        fx: float = 3000.0,
        fy: float = 3000.0,
        initial_position: np.ndarray | None = None,
        initial_look_at: np.ndarray | None = None,
        batch_size: int = 10000,
        debug: bool = False,
    ) -> None:
        
        # Viewer parameters
        self.batch_size = batch_size
        self.debug = debug
        
        # Camera parameters
        self.w = width
        self.h = height
        self.fx = fx
        self.fy = fy
        self.camera = PinholeCamera(fx, fy, width / 2, height / 2, debug=debug)
        
        # Load pointcloud
        self.pointcloud = EuclideanPointcloud.from_ply(ply_path)
        if debug:
            print(f"Loaded pointcloud with {len(self.pointcloud.points)} points from {ply_path}")
            
        # Mouse interaction state
        self.is_dragging = False
        self.is_rotating = False
        self.last_x: float | None = None
        self.last_y: float | None = None
        self.anchor_point = None
        self.anchor_point_u_v  = None

        # Initial camera setup
        if initial_position is not None:
            self.camera.set_position(EuclideanPoint(initial_position))
        else:
            self.camera.set_position(EuclideanPoint(np.array([0.0, 0.0, -1.0])))
        
        if initial_look_at is not None:
            self.camera.look_at(EuclideanPoint(initial_look_at))
        else:
            self.camera.look_at(EuclideanPoint(np.array([0.0, 0.0, 1.0])))
  
        f, u, v = self.camera.get_forward_up_right()
        if self.debug:
            print("Initial camera axes:")
            print(f"forward: {f}")
            print(f"up:      {u}")
            print(f"right:   {v}")
            
        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.render()
        self.im = self.ax.imshow(self.image, cmap="gray")
        plt.axis("off")

        # Connect events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        
    def update_camera_from_angles(self, dyaw: float, dpitch: float) -> None:
        """ Implements camera rotation around an anchor point using quaternions. 
        This control mechanism is known as "arcball" rotation.
        It keeps the anchor point fixed in the view while rotating the camera around it for smooth interaction.
         Args:
            dyaw: change in yaw angle (radians)
            dpitch: change in pitch angle (radians)
        """
        forward, up, right = self.camera.get_forward_up_right()
        P = self.camera.get_orientation_position()[1]
        V = P.x - self.anchor_point.x
        
        # Find quaternion for combined rotation
        q_h = np.array([np.cos(dyaw / 2), *(up * np.sin(dyaw / 2))]) # quaternion for rotating around up (dyaw radians)
        q_v = np.array([np.cos(dpitch / 2), *(right * np.sin(dpitch / 2))]) # quaternion for rotating around right (dpitch radians)
        q = multiply_quaternions(q_h, q_v) # Hamilton product
        
        # Rotate camera position 
        V_rotated = rotate_vector_by_quaternion(q, V)
        P_new = self.anchor_point.x + V_rotated
        self.camera.set_position(EuclideanPoint(P_new)) # new position
        
        # Rotate orientation
        forward_rotated = rotate_vector_by_quaternion(q, forward)
        up_rotated = rotate_vector_by_quaternion(q, up)
        self.camera.look_at(EuclideanPoint(P_new + forward_rotated), up_rotated)
        
        R = self.camera.get_rotation_translation()[0].mat
        if self.debug: assert abs((np.linalg.inv(R) - R.T).sum()) < 1e-6, "Rotation matrix is not orthogonal" 
        
        # Adjust camera position to keep anchor point fixed in view:
        # fix (u,v) -> direction vector in camera coords --> direction vector in world coords --> new camera position
        K = self.camera.get_intrinsics().mat
        K_inv = np.linalg.inv(K)
        D_cam = K_inv @ np.array([self.anchor_point_u_v[0], self.anchor_point_u_v[1], 1.0])
        D_world = R.T @ D_cam # R^inv = R^T for rotation matrices
        P_new = self.anchor_point.x - D_world
        self.camera.set_position(EuclideanPoint(P_new))        
                              
    def update_camera_from_translation(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        """Translate the camera in its local right and up directions.
        Args:
            dx: translation along the camera's right axis
            dy: translation along the camera's up axis (downwards)
            dz: translation along the camera's forward axis
        """
        forward, up, right = self.camera.get_forward_up_right()
        P = self.camera.get_orientation_position()[1]
        translation = right * dx + up * dy + forward * dz 
        P_new = P.x + translation
        self.camera.set_position(EuclideanPoint(P_new))                        
                                              
    def capture(self) -> np.ndarray:
        """Capture the current view of the pointcloud as projected points array."""
        projected_points = self.camera(
            self.pointcloud,
            self.h,
            self.w,
            batch_size=self.batch_size,
            background=None
        )
        return projected_points

    def render(self) -> None:
        """Render an image from the current projected points."""
        self.projected_points = self.capture()
        mask = (self.projected_points != None)
        self.image = np.zeros(mask.shape, dtype=np.uint8)
        self.image[mask] = 255

    # Event handlers
    def on_key(self, event) -> None:  # type: ignore[override]
        """Handle key press events to translate the camera.
        Note: the camera actually moves in opposite direction to give the effect of moving the view.
        """
        step_x, step_y = 0.1 * self.w / self.fx, 0.1 * self.h / self.fy
        dx, dy = 0.0, 0.0
        if event.key == "up":
            dy = step_y
        elif event.key == "down":
            dy = -step_y
        elif event.key == "left":
            dx = step_x
        elif event.key == "right":
            dx = -step_x
        else:
            if self.debug: print(event.key)
            return
        
        self.update_camera_from_translation(dx, dy)

        self.render()
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event) -> None:  # type: ignore[override]
        """Handle scroll events to move the camera forward/backward."""
        step_z = 0.1
        if event.button == "up":  # wheel up
            dz = step_z
        elif event.button == "down":  # wheel down
            dz = -step_z

        self.update_camera_from_translation(dz=dz)

        self.render()
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def on_button_press(self, event) -> None:  # type: ignore[override]
        if event.button == 1: # left click anchors a rotation point
            # Convert from display coords (pixels on the window) to data coords
            self.is_rotating = True
            self.last_x, self.last_y = event.x, event.y
            
        if event.button == 2:  # middle click 
            self.is_dragging = True
            self.last_x, self.last_y = event.x, event.y

        if event.button == 3:  # right click
            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None:
                return
            u = int(round(xdata))  # column
            v = int(round(ydata))  # row
            anchor_point = self.projected_points[v, u]
            if anchor_point != None:
                print(f"Set point {anchor_point.x} as anchor\n")
                self.anchor_point = anchor_point
                self.anchor_point_u_v = (u, v)
        
    def on_button_release(self, event) -> None:  # type: ignore[override]
        if event.button == 1:
            self.is_rotating = False
            
        if event.button == 2:
            self.is_dragging = False
            
    def on_motion(self, event) -> None:  # type: ignore[override]
        if (
            not (self.is_dragging or self.is_rotating)
            or event.x is None
            or event.y is None
            or self.last_x is None
            or self.last_y is None
        ):
            return

        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y
        if self.debug: print(f"Dragging to ({event.x}, {event.y}), delta=({dx}, {dy})")

        if self.is_rotating:
            self.update_camera_from_angles(dx * 0.01, dy * 0.01)
            
        if self.is_dragging:
            if self.debug: print("Dragging failed")

        self.render()
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()

