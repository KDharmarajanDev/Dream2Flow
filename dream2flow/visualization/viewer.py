import viser
import torch
import numpy as np
from typing import Optional, Dict, Tuple
from dream2flow.flow.object_flow import ObjectFlow

class Dream2FlowViewer:
    """
    Viser-based 3D visualization for Dream2Flow.
    """
    
    _instance = None

    def __init__(self, port: int = 8080):
        if Dream2FlowViewer._instance is not None:
            raise RuntimeError("Dream2FlowViewer is a singleton. Use get_viewer() instead.")
        
        self.server = viser.ViserServer(port=port)
        self._cameras = {}
        Dream2FlowViewer._instance = self

    @classmethod
    def get_viewer(cls, port: int = 8080) -> 'Dream2FlowViewer':
        if cls._instance is None:
            cls(port=port)
        return cls._instance

    def register_camera(self, name: str, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        """
        Register a camera for visualization.
        """
        # extrinsics is (4,4) [R | T]
        # Viser expects wxyz quaternion and position
        
        # We can visualize a frustum or just store it
        self._cameras[name] = (intrinsics, extrinsics)
        
        # Visualize coordinate frame for camera
        pos = extrinsics[:3, 3].detach().cpu().numpy()
        rot_mat = extrinsics[:3, :3].detach().cpu().numpy()
        
        # Convert rot_mat to wxyz quat
        from scipy.spatial.transform import Rotation as R
        quat_xyzw = R.from_matrix(rot_mat).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        self.server.scene.add_frame(
            f"/cameras/{name}",
            wxyz=quat_wxyz,
            position=pos,
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.005
        )

    def visualize_point_cloud(self, name: str, points: torch.Tensor, colors: Optional[torch.Tensor] = None, point_size: float = 0.005):
        """
        Visualize a 3D point cloud.
        """
        points_np = points.detach().cpu().numpy()
        if colors is not None:
            colors_np = colors.detach().cpu().numpy()
        else:
            colors_np = np.ones_like(points_np) * 0.5 # Default gray
            
        self.server.scene.add_point_cloud(
            name=name,
            points=points_np,
            colors=colors_np,
            point_size=point_size
        )

    def visualize_object_flow(self, flow: ObjectFlow, name: str, show_all_timesteps: bool = False, point_size: float = 0.005):
        """
        Visualize an ObjectFlow object.
        """
        T = flow.position.shape[0]
        
        if show_all_timesteps:
            # Flatten all timesteps
            valid_mask = flow.valid_mask.reshape(-1)
            positions = flow.position.reshape(-1, 3)[valid_mask]
            if flow.rgb is not None:
                colors = flow.rgb.reshape(-1, 3)[valid_mask]
            else:
                colors = None
            self.visualize_point_cloud(f"{name}/all_points", positions, colors, point_size)
        else:
            # Create a slider for timesteps
            slider = self.server.gui.add_slider(
                f"{name}/timestep",
                min=0,
                max=T - 1,
                step=1,
                initial_value=0
            )
            
            @slider.on_update
            def _(_):
                t = slider.value
                valid_mask = flow.valid_mask[t]
                positions = flow.position[t][valid_mask]
                if flow.rgb is not None:
                    colors = flow.rgb[t][valid_mask]
                else:
                    colors = None
                self.visualize_point_cloud(f"{name}/points", positions, colors, point_size)
            
            # Trigger initial update
            _ (None)

    def visualize_lines(self, name: str, points: torch.Tensor, color: torch.Tensor, line_width: float = 2.0):
        """
        Visualize line segments.
        """
        points_np = points.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        
        self.server.scene.add_line_segments(
            name=name,
            points=points_np,
            colors=color_np,
            line_width=line_width
        )

    def visualize_batched_axes(self, name: str, positions: torch.Tensor, rotations: torch.Tensor, scale: float = 0.05):
        """
        Visualize a sequence of coordinate frames (e.g. for a trajectory).
        rotations are xyzw quaternions.
        """
        pos_np = positions.detach().cpu().numpy()
        rot_np = rotations.detach().cpu().numpy()
        
        for i in range(pos_np.shape[0]):
            # Convert xyzw to wxyz
            quat_wxyz = np.array([rot_np[i, 3], rot_np[i, 0], rot_np[i, 1], rot_np[i, 2]])
            self.server.scene.add_frame(
                f"{name}/{i}",
                wxyz=quat_wxyz,
                position=pos_np[i],
                show_axes=True,
                axes_length=scale,
                axes_radius=scale/10.0
            )

    def visualize_workspace_bounds(self, name: str, bounds: np.ndarray, color: Tuple[int, int, int] = (200, 200, 200)):
        """
        Visualize workspace bounds as a wireframe box.
        bounds is [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        z_min, z_max = bounds[2]
        
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ])
        
        edges = [
            (0,1), (1,2), (2,3), (3,0), # bottom
            (4,5), (5,6), (6,7), (7,4), # top
            (0,4), (1,5), (2,6), (3,7)  # vertical
        ]
        
        line_points = []
        for start, end in edges:
            line_points.append(corners[start])
            line_points.append(corners[end])
            
        line_points = np.array(line_points)
        colors = np.tile(np.array(color) / 255.0, (len(edges), 1))
        
        self.server.scene.add_line_segments(
            name=name,
            points=line_points,
            colors=colors,
            line_width=1.0
        )
