import torch
import os
from typing import Optional, Dict
from dream2flow.flow.generators.base import ObjectFlowGenerator
from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.flow.object_flow import ObjectFlow
from dream2flow.camera import CameraCalibration

class VideoFlowGenerator(ObjectFlowGenerator):
    """
    Generates 3D object flow from video frames, depth frames, and 2D tracks.
    """

    def generate(self, 
                 output_dir: str, 
                 start_image: torch.Tensor, 
                 camera: CameraCalibration,
                 camera_name: str,
                 depth_frames: torch.Tensor,
                 pixel_coords_2d: torch.Tensor,
                 tracks_2d: torch.Tensor,
                 **kwargs) -> ObjectFlowResult:
        """
        Generates 3D object flow by lifting 2D tracks to 3D using depth.

        Args:
            output_dir: Directory to save results
            start_image: Initial RGB image (H, W, 3)
            camera: Camera calibration information
            camera_name: Name of the camera being used
            depth_frames: Depth frames (T, H, W) in meters
            pixel_coords_2d: Image coordinates of points in initial frame (N, 2)
            tracks_2d: 2D pixel trajectories (T, N, 2)
            **kwargs: Additional arguments
            
        Returns:
            ObjectFlowResult: Result of object flow generation
        """
        T, N, _ = tracks_2d.shape
        device = tracks_2d.device
        
        intrinsics, extrinsics = camera.get_camera_calibration(camera_name)
        intrinsics = intrinsics.to(device)
        extrinsics = extrinsics.to(device)
        
        # Lift 2D tracks to 3D
        # tracks_2d: (T, N, 2) -> (x, y)
        # depth_frames: (T, H, W)
        
        # Get depth for each point at each timestep
        # We need to sample depth_frames at tracks_2d coordinates
        # tracks_2d are (x, y) coordinates. depth_frames is indexed by (y, x).
        
        t_indices = torch.arange(T, device=device).view(T, 1).repeat(1, N)
        y_coords = tracks_2d[..., 1].long().clamp(0, depth_frames.shape[1] - 1)
        x_coords = tracks_2d[..., 0].long().clamp(0, depth_frames.shape[2] - 1)
        
        point_depths = depth_frames[t_indices, y_coords, x_coords] # (T, N)
        
        # Back-project to 3D camera coordinates
        # x_cam = (x_img - cx) * depth / fx
        # y_cam = (y_img - cy) * depth / fy
        # z_cam = depth
        
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        x_img = tracks_2d[..., 0]
        y_img = tracks_2d[..., 1]
        
        x_cam = (x_img - cx) * point_depths / fx
        y_cam = (y_img - cy) * point_depths / fy
        z_cam = point_depths
        
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1) # (T, N, 3)
        
        # Transform to world coordinates
        # P_world = R * P_cam + T
        # extrinsics is [R | T] (4, 4)
        
        points_cam_homo = torch.cat([points_cam, torch.ones_like(points_cam[..., :1])], dim=-1) # (T, N, 4)
        points_world_homo = torch.einsum('ij,tnj->tni', extrinsics, points_cam_homo)
        points_world = points_world_homo[..., :3] # (T, N, 3)
        
        # Create ObjectFlow
        flow = ObjectFlow(
            position=points_world,
            valid_mask=(point_depths > 0)
        )
        flow.initialize_color()
        
        result = ObjectFlowResult(
            flow=flow,
            initial_pixel_coords=pixel_coords_2d,
            pixel_trajectories=tracks_2d
        )
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            result.save(os.path.join(output_dir, "object_flow_result.pt"))
            
        return result
