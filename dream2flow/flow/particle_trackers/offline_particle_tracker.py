from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.functional import grid_sample

from dream2flow.camera import CameraCalibration
from dream2flow.flow.geometry import depth_to_world_points
from dream2flow.flow.object_flow import ObjectFlow
from dream2flow.flow.particle_trackers.base import ParticleTracker, ParticleTrackingResult


@dataclass
class OfflineParticleTrackerConfig:
    visualize_tracking: bool = False
    pad_value: int = 0
    tracks_leave_trace: int = 30
    visualization_fps: int = 24


class OfflineParticleTracker(ParticleTracker):
    def __init__(self, config: OfflineParticleTrackerConfig, device: str = "cuda") -> None:
        self.config = config
        self._device = device
        self._offline_cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self._device)
        self._query_points: torch.Tensor | None = None

    def initialize_tracking(self, query_points: torch.Tensor) -> None:
        if query_points.dim() != 2 or query_points.shape[1] != 3:
            raise ValueError(f"query_points must be of shape (N, 3), got {tuple(query_points.shape)}")
        self._query_points = query_points.unsqueeze(0).float()

    def _get_tracked_depth_values(
        self,
        depth_images: torch.Tensor,
        valid_estimated_depth_mask: torch.Tensor,
        tracks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = depth_images.shape
        normalized_tracks = tracks.clone().unsqueeze(1)
        normalized_tracks[..., 0] = (normalized_tracks[..., 0] / width) * 2 - 1
        normalized_tracks[..., 1] = (normalized_tracks[..., 1] / height) * 2 - 1
        data_for_interpolation = torch.cat((depth_images, valid_estimated_depth_mask.float()), dim=1)
        interpolated_data = grid_sample(data_for_interpolation, normalized_tracks, align_corners=True)
        interpolated_depth_values = interpolated_data[:, 0].squeeze(1)
        interpolated_valid_mask = interpolated_data[:, 1].squeeze(1) == 1
        return interpolated_depth_values, interpolated_valid_mask

    def track_particles(
        self,
        output_dir: str,
        camera: CameraCalibration,
        camera_name: str,
        rgb_images: torch.Tensor,
        depth_images: torch.Tensor,
        valid_estimated_depth_mask: torch.Tensor,
    ) -> ParticleTrackingResult:
        if self._query_points is None:
            raise RuntimeError("Tracking cannot be initialized without query points")

        batched_rgb_images = rgb_images.unsqueeze(0)
        predicted_tracks, predicted_visibilities = self._offline_cotracker(
            video=batched_rgb_images,
            queries=self._query_points,
        )
        non_batched_predicted_tracks = predicted_tracks.squeeze(0)

        tracked_depth_values, valid_track_mask = self._get_tracked_depth_values(
            depth_images=depth_images.unsqueeze(1),
            valid_estimated_depth_mask=valid_estimated_depth_mask.unsqueeze(1),
            tracks=non_batched_predicted_tracks,
        )
        valid_track_mask &= predicted_visibilities.squeeze(0)

        intrinsics, extrinsics = camera.get_camera_calibration(camera_name)
        intrinsics = intrinsics.to(depth_images.device)
        extrinsics = extrinsics.to(depth_images.device)
        particle_position_trajectory = depth_to_world_points(
            tracked_depth_values,
            intrinsics,
            extrinsics,
            pixel_coords=non_batched_predicted_tracks,
        )

        flow = ObjectFlow(
            position=particle_position_trajectory,
            valid_mask=valid_track_mask,
        )
        flow.initialize_color()

        if self.config.visualize_tracking:
            from cotracker.utils.visualizer import Visualizer

            visualizer = Visualizer(
                save_dir=str(Path(output_dir)),
                pad_value=self.config.pad_value,
                linewidth=3,
                fps=self.config.visualization_fps,
                mode="rainbow",
                tracks_leave_trace=self.config.tracks_leave_trace,
            )
            visualizer.visualize(
                batched_rgb_images,
                predicted_tracks,
                predicted_visibilities,
                filename="track_vis",
            )

        return ParticleTrackingResult(
            flow=flow,
            particle_pixel_trajectories=non_batched_predicted_tracks,
            predicted_visibilities=predicted_visibilities.squeeze(0),
        )
