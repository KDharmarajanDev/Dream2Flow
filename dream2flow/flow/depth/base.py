from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class DepthExtractorConfig:
    resize_height: int = 720
    resize_width: int = 1280
    use_affine_calibration: bool = False


class DepthExtractor(ABC):
    def __init__(self, config: DepthExtractorConfig, device: str) -> None:
        self.config = config
        self.device = device

    @abstractmethod
    def extract_depth_from_file(self, scene_dir: str, video_path: str) -> torch.Tensor:
        pass

    @abstractmethod
    def extract_depth_from_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        pass

    def calibrate_depth(
        self,
        extracted_depth: torch.Tensor,
        ground_truth_depth: torch.Tensor,
        calibration_mask: torch.Tensor,
    ) -> torch.Tensor:
        if extracted_depth.ndim == 4:
            extracted_depth = extracted_depth.squeeze(1)

        ground_truth_depth = ground_truth_depth[calibration_mask]
        initial_extracted_depth = extracted_depth[0][calibration_mask]

        valid_mask = initial_extracted_depth > 0
        initial_extracted_depth = initial_extracted_depth[valid_mask]
        ground_truth_depth = ground_truth_depth[valid_mask]

        if self.config.use_affine_calibration:
            matrix = torch.stack(
                [
                    initial_extracted_depth,
                    torch.ones_like(initial_extracted_depth, device=self.device),
                ],
                dim=-1,
            )
            target = ground_truth_depth
            scale_shift = torch.linalg.lstsq(matrix, target).solution
            scale, shift = scale_shift[0], scale_shift[1]
        else:
            matrix = initial_extracted_depth.unsqueeze(1)
            target = ground_truth_depth
            scale_shift = torch.linalg.lstsq(matrix, target).solution
            scale, shift = scale_shift[0], 0

        print(f"Scale: {scale}, Shift: {shift}")
        return scale * extracted_depth + shift
