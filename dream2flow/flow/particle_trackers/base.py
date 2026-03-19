from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from dream2flow.camera import CameraCalibration
from dream2flow.flow.object_flow import ObjectFlow


@dataclass
class ParticleTrackingResult:
    flow: ObjectFlow | None = None
    particle_pixel_trajectories: torch.Tensor | None = None
    predicted_visibilities: torch.Tensor | None = None


class ParticleTracker(ABC):
    @abstractmethod
    def initialize_tracking(self, query_points: torch.Tensor) -> None:
        pass

    @abstractmethod
    def track_particles(
        self,
        output_dir: str,
        camera: CameraCalibration,
        camera_name: str,
        rgb_images: torch.Tensor,
        depth_images: torch.Tensor,
        valid_estimated_depth_mask: torch.Tensor,
    ) -> ParticleTrackingResult:
        pass
