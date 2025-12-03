import time

import matplotlib.pyplot as plt
import numpy as np

from camera_geometry.camera import PinholeCamera
from camera_geometry.euclidean_space import EuclideanPoint
from camera_geometry.projective_space import ProjectivePointcloud
from camera_geometry.vector_ops import multiply_quaternions


class PointcloudViewer:
    def __init__(
        self,
        ply_path: str = "scripts/dragon_vrip_res2.ply",
        width: int = 1000,
        height: int = 1000,
        fx: float = 3000.0,
        fy: float = 3000.0,
        debug: bool = True,
    ) -> None:
        # Load pointcloud
        self.pointcloud = ProjectivePointcloud.from_ply(ply_path)
        print(len(self.pointcloud))
        self.anchor_point = self.pointcloud.points[0] #! to change

        # Camera parameters
        self.w = width
        self.h = height
        self.camera = PinholeCamera(fx, fy, width / 2, height / 2, debug=debug)

        # Mouse interaction state
        self.is_dragging = False
        self.last_x: float | None = None
        self.last_y: float | None = None

        # Initial camera setup
        self.camera.set_position(EuclideanPoint(np.array([0.0, 0.0, -1.0])))
        self.camera.look_at(EuclideanPoint(np.array([0.0, 0.0, 1.0])))
  
        f, u, v = self.camera.get_forward_up_right()
        print("Initial camera axes:")
        print(f)
        print(u)
        print(v)

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

    def update_camera_from_angles_before(self, dyaw: float, dpitch: float) -> None:
        print(f"Updating camera with dyaw={dyaw}, dpitch={dpitch}")
        f, u, v = self.camera.get_forward_up_right()
        new_forward_u = f * np.cos(dpitch) + u * np.sin(dpitch)
        new_forward_v = f * np.cos(-dyaw) + v * np.sin(-dyaw)
        new_forward = new_forward_u + new_forward_v
        new_forward = new_forward / np.linalg.norm(new_forward)
        print(new_forward)
        print()

        _, pos = self.camera.get_orientation_position()
        self.camera.look_at(pos + new_forward)
        
    def update_camera_from_angles(self, dyaw: float, dpitch: float) -> None:
        forward, up, right = self.camera.get_forward_up_right()
        P = self.camera.get_orientation_position()[1]
        V = P.x - self.anchor_point.x
        
        q_h = np.array([np.cos(dyaw / 2), *(up * np.sin(dyaw / 2))])
        q_v = np.array([np.cos(dpitch / 2), *(right * np.sin(dpitch / 2))])
        # Hamilton product: q_h * q_v
        q = multiply_quaternions(q_h, q_v)
        # Rotate V by conjugating: q * V * q^*
        # q^* (conjugate) = [w, -x, -y, -z]
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        V_quat = np.array([0.0, *V])
        V_rotated_quat = multiply_quaternions(multiply_quaternions(q, V_quat), q_conj)
        V_rotated = V_rotated_quat[1:]  # extract vector part
        P_new = self.anchor_point.x + V_rotated

        self.camera.set_position(EuclideanPoint(P_new))
        self.camera.look_at(EuclideanPoint(self.anchor_point.x))

                                                                                
    def capture(self) -> None:
        """Capture the current view of the pointcloud as projected points array."""
        start = time.time()
        projected_points = self.camera(
            self.pointcloud,
            self.h,
            self.w,
            batch_size=10000,
            background=None
        )
        print(f"Rendering array took {time.time() - start:.4f} s")
        return projected_points

    def render(self) -> None:
        """Render an image from the current projected points."""
        self.projected_points = self.capture()
        mask = (self.projected_points != None)
        self.image = np.zeros(mask.shape, dtype=np.uint8)
        self.image[mask] = 255

    # Event handlers
    def on_key(self, event) -> None:  # type: ignore[override]
        step = 0.1
        _, C = self.camera.get_rotation_translation()  # C is EuclideanPoint
        pos = C.x.copy()  # numpy array (3,)

        if event.key == "up":
            pos[1] -= step
        elif event.key == "down":
            pos[1] += step
        elif event.key == "left":
            pos[0] -= step
        elif event.key == "right":
            pos[0] += step
        else:
            print(event.key)

        self.camera.set_position(EuclideanPoint(pos))
        self.camera.look_at(EuclideanPoint(np.array([0.0, 0.0, 1.0])))

        self.render()
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event) -> None:  # type: ignore[override]
        step = 0.2  # how much to move per wheel tick
        _, C = self.camera.get_rotation_translation()
        pos = C.x.copy()  # numpy array (3,)

        if event.button == "up":  # wheel up
            pos[2] += step  # move camera forward/back in z
        elif event.button == "down":  # wheel down
            pos[2] -= step

        self.camera.set_position(EuclideanPoint(pos))
        self.camera.look_at(EuclideanPoint(np.array([0.0, 0.0, 1.0])))

        self.render()
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def on_button_press(self, event) -> None:  # type: ignore[override]
        if event.button == 1:  # left click
            self.is_dragging = True
            self.last_x, self.last_y = event.x, event.y
            print(f"Started dragging at ({self.last_x}, {self.last_y})")
            clicked_pt = self.projected_points[self.last_y, self.last_x]
            if clicked_pt != None:
                print(f"Clicked on the point {clicked_pt}")
                self.anchor_point = clicked_pt
        
        if event.button == 3:  # right click anchors a rotation point
            anchor_point = self.projected_points[event.y, event.x]
            if anchor_point != None:
                print(f"Clicked on the point {anchor_point}")
                self.anchor_point = anchor_point
                

    def on_button_release(self, event) -> None:  # type: ignore[override]
        if event.button == 1:
            self.is_dragging = False
            print("Stopped dragging")

    def on_motion(self, event) -> None:  # type: ignore[override]
        if (
            not self.is_dragging
            or event.x is None
            or event.y is None
            or self.last_x is None
            or self.last_y is None
        ):
            return

        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.last_x, self.last_y = event.x, event.y
        print(f"Dragging to ({event.x}, {event.y}), delta=({dx}, {dy})")

        self.update_camera_from_angles(dx * 0.005, dy * 0.005)

        self.render()
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()
        input("Press Enter to exit...")


if __name__ == "__main__":
    viewer = PointcloudViewer()
    viewer.show()