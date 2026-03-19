from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from dream2flow.flow.generators.base import ObjectFlowGenerator
from dream2flow.flow.depth import DepthExtractor
from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.flow.geometry import depth_to_world_points
from dream2flow.flow.particle_trackers import OfflineParticleTracker, OfflineParticleTrackerConfig
from dream2flow.flow.region_selectors import GroundedRegionSelector, GroundedRegionSelectorConfig
from dream2flow.camera import CameraCalibration


@dataclass
class VideoFlowGeneratorConfig:
    save_intermediate_results: bool = True
    particle_ratio: float = 0.01
    max_num_particles: int = 1024
    mask_erosion_kernel_size: int = 11
    tracker: OfflineParticleTrackerConfig = field(default_factory=OfflineParticleTrackerConfig)
    region_selector: GroundedRegionSelectorConfig = field(default_factory=GroundedRegionSelectorConfig)


class VideoFlowGenerator(ObjectFlowGenerator):
    """
    Generates 3D object flow using the same high-level sequence as the
    reference video-based particle trajectory generator.
    """

    def __init__(
        self,
        depth_extractor: DepthExtractor,
        device: str = "cpu",
        config: VideoFlowGeneratorConfig | None = None,
        region_selector: GroundedRegionSelector | None = None,
        particle_tracker: OfflineParticleTracker | None = None,
    ) -> None:
        self.device = device
        self.config = config or VideoFlowGeneratorConfig()
        self._depth_extractor = depth_extractor
        self._region_selector = region_selector or GroundedRegionSelector(self.config.region_selector, device=device)
        self._particle_tracker = particle_tracker or OfflineParticleTracker(self.config.tracker, device=device)

    def _save_rgb_if_needed(self, output_dir: Path, start_image: torch.Tensor) -> None:
        if not self.config.save_intermediate_results:
            return
        rgb_save_file_path = output_dir / "camera_rgb.png"
        if rgb_save_file_path.exists():
            return
        rgb_img = start_image.detach().cpu().numpy()
        cv2.imwrite(str(rgb_save_file_path), cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    def _save_depth_if_needed(self, output_dir: Path, initial_depth: torch.Tensor) -> None:
        if not self.config.save_intermediate_results:
            return
        initial_depth_file_path = output_dir / "camera_depth.npy"
        if not initial_depth_file_path.exists():
            np.save(initial_depth_file_path, initial_depth.detach().cpu().numpy())

    def _save_camera_if_needed(self, output_dir: Path, camera: CameraCalibration) -> None:
        if not self.config.save_intermediate_results:
            return
        camera_calibration_info_file_path = output_dir / "camera_calibration_info.json"
        if not camera_calibration_info_file_path.exists():
            camera.save(str(camera_calibration_info_file_path))

    def _erode_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        kernel = torch.ones((kernel_size, kernel_size), device=mask.device)
        eroded_mask = torch.nn.functional.conv2d(
            mask.float().unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=kernel_size // 2,
        ).squeeze() == kernel.sum()
        return eroded_mask

    def _fps(self, object_points: torch.Tensor, ratio: float) -> torch.Tensor:
        from torch_geometric.nn import fps

        return fps(object_points, ratio=ratio)

    def _get_tracked_particle_image_coordinates(
        self,
        output_dir: Path,
        instruction: str,
        initial_rgb_img_torch: torch.Tensor,
        initial_depth_arr_torch: torch.Tensor,
        valid_estimated_depth_mask: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tracked_object_mask = self._region_selector.extract_region(
            output_dir=str(output_dir),
            image=initial_rgb_img_torch,
            instruction=instruction,
        )
        tracked_object_mask = self._erode_mask(
            tracked_object_mask,
            kernel_size=self.config.mask_erosion_kernel_size,
        ) & valid_estimated_depth_mask

        mask_indices = torch.nonzero(tracked_object_mask)
        mask_pixel_coords = torch.stack((mask_indices[:, 1], mask_indices[:, 0]), dim=1)
        object_points = depth_to_world_points(
            depth=initial_depth_arr_torch[tracked_object_mask].unsqueeze(0),
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            pixel_coords=mask_pixel_coords.unsqueeze(0),
        ).squeeze(0)
        if object_points.shape[0] == 0:
            raise RuntimeError("The grounded object mask is empty after erosion and depth filtering.")

        particle_ratio = min(
            self.config.particle_ratio,
            self.config.max_num_particles / object_points.shape[0],
        )
        kept_object_points_idx = self._fps(object_points, ratio=particle_ratio)
        kept_object_pixel_coords = mask_pixel_coords[kept_object_points_idx]
        return kept_object_pixel_coords, tracked_object_mask

    def _get_calibrated_depth_frames(
        self,
        depth_frames: torch.Tensor,
        initial_depth_arr_torch: torch.Tensor,
        object_mask: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        calibration_mask = object_mask
        if depth_valid_mask is not None:
            calibration_mask = calibration_mask & depth_valid_mask

        return self._depth_extractor.calibrate_depth(
            extracted_depth=depth_frames,
            ground_truth_depth=initial_depth_arr_torch,
            calibration_mask=calibration_mask,
        )

    def generate(
        self,
        output_dir: str,
        start_image: torch.Tensor,
        camera: CameraCalibration,
        camera_name: str,
        instruction: str,
        video_frames: torch.Tensor,
        initial_depth: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
        video_path: str = "rgb.mp4",
        **kwargs,
    ) -> ObjectFlowResult:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        initial_depth = initial_depth.to(dtype=torch.float32, device=self.device)

        self._save_rgb_if_needed(output_dir_path, start_image)
        self._save_depth_if_needed(output_dir_path, initial_depth)
        self._save_camera_if_needed(output_dir_path, camera)

        depth_frames = self._depth_extractor.extract_depth_from_file(output_dir, video_path)
        depth_frames = depth_frames.to(dtype=torch.float32, device=self.device)
        if self.config.save_intermediate_results:
            depth_save_path = output_dir_path / "depth_frames.pt"
            if not depth_save_path.exists():
                torch.save(depth_frames.detach().cpu(), depth_save_path)

        video_frames = video_frames.to(self.device)
        intrinsics, extrinsics = camera.get_camera_calibration(camera_name)
        intrinsics = intrinsics.to(self.device)
        extrinsics = extrinsics.to(self.device)

        tracked_object_pixel_coords, tracked_region_mask = self._get_tracked_particle_image_coordinates(
            output_dir=output_dir_path,
            instruction=instruction,
            initial_rgb_img_torch=start_image,
            initial_depth_arr_torch=initial_depth,
            valid_estimated_depth_mask=depth_frames[0] > 0,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
        )

        tracker_query_points = torch.stack(
            (
                torch.zeros_like(tracked_object_pixel_coords[:, 0]),
                tracked_object_pixel_coords[:, 0],
                tracked_object_pixel_coords[:, 1],
            ),
            dim=1,
        )
        self._particle_tracker.initialize_tracking(tracker_query_points)

        calibrated_depth_frames = self._get_calibrated_depth_frames(
            depth_frames=depth_frames,
            initial_depth_arr_torch=initial_depth,
            object_mask=tracked_region_mask,
            depth_valid_mask=depth_valid_mask,
        )

        restructured_video_frames = video_frames.float().permute(0, 3, 1, 2)
        tracking_result = self._particle_tracker.track_particles(
            output_dir=output_dir,
            camera=camera,
            camera_name=camera_name,
            rgb_images=restructured_video_frames,
            depth_images=calibrated_depth_frames,
            valid_estimated_depth_mask=depth_frames > 0,
        )
        flow = tracking_result.flow
        if flow is None:
            raise RuntimeError("Particle tracker did not return a 3D flow")

        tracks_path = output_dir_path / "tracks_2d.pt"
        if self.config.save_intermediate_results and not tracks_path.exists():
            torch.save(tracking_result.particle_pixel_trajectories.detach().cpu(), tracks_path)

        result = ObjectFlowResult(
            flow=flow,
            initial_pixel_coords=tracked_object_pixel_coords,
            region_mask=tracked_region_mask,
            pixel_trajectories=tracking_result.particle_pixel_trajectories,
        )
        return result
