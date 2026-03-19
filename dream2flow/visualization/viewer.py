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
        self._camera_name_to_frustum_handle = {}
        self._point_cloud_name_to_handle = {}
        Dream2FlowViewer._instance = self

    @classmethod
    def get_viewer(cls, port: int = 8080) -> 'Dream2FlowViewer':
        if cls._instance is None:
            cls(port=port)
        return cls._instance

    def _mat_to_viser_pose(self, extrinsic_matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        position = extrinsic_matrix[:3, 3].detach().cpu().numpy()

        from scipy.spatial.transform import Rotation as R

        xyzw = R.from_matrix(extrinsic_matrix[:3, :3].detach().cpu().numpy()).as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        return position, wxyz

    def register_camera(self, name: str, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        """
        Register a camera for visualization.
        """
        self._cameras[name] = (intrinsics, extrinsics)

        intrinsic_matrix = intrinsics.detach().cpu().numpy()
        width = max(float(intrinsic_matrix[0, 0]), 1.0)
        height = max(float(intrinsic_matrix[1, 1]), 1.0)
        fy = float(intrinsic_matrix[1, 1])
        position, wxyz = self._mat_to_viser_pose(extrinsics)
        fov = 2.0 * np.arctan2(height / 2.0, fy) if fy != 0 else np.deg2rad(60.0)
        aspect = width / height

        if hasattr(self.server.scene, "add_camera_frustum"):
            if name not in self._camera_name_to_frustum_handle:
                self._camera_name_to_frustum_handle[name] = self.server.scene.add_camera_frustum(
                    f"/cameras/{name}/frustum",
                    fov=fov,
                    aspect=aspect,
                    scale=0.05,
                    wxyz=wxyz,
                    position=position,
                )
            frustum_handle = self._camera_name_to_frustum_handle[name]
            frustum_handle.wxyz = wxyz
            frustum_handle.position = position
            frustum_handle.fov = fov
            frustum_handle.aspect = aspect

    def visualize_point_cloud(self, name: str, points: torch.Tensor, colors: Optional[torch.Tensor] = None, point_size: float = 0.005):
        """
        Visualize a 3D point cloud.
        """
        points_np = points.detach().cpu().numpy()
        if colors is not None:
            colors_np = colors.detach().cpu().numpy()
        else:
            colors_np = np.ones_like(points_np) * 0.5 # Default gray
            
        if name not in self._point_cloud_name_to_handle:
            self._point_cloud_name_to_handle[name] = self.server.scene.add_point_cloud(
                name=name,
                points=points_np,
                colors=colors_np,
                point_shape="circle",
                point_size=point_size
            )
        point_cloud_handle = self._point_cloud_name_to_handle[name]
        point_cloud_handle.points = points_np
        point_cloud_handle.colors = colors_np
        point_cloud_handle.point_size = point_size

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
