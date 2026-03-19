from dataclasses import dataclass
from pathlib import Path

import torch

from dream2flow.flow.depth.base import DepthExtractor, DepthExtractorConfig


@dataclass
class FileDepthExtractorConfig(DepthExtractorConfig):
    depth_file_path: str = "depth_frames.pt"


class FileDepthExtractor(DepthExtractor):
    def __init__(self, config: FileDepthExtractorConfig, device: str) -> None:
        super().__init__(config, device)
        self.depth_data: torch.Tensor | None = None

    def extract_depth_from_file(self, scene_dir: str, video_path: str) -> torch.Tensor:
        depth_file_path = Path(scene_dir) / self.config.depth_file_path
        depth_data = torch.load(depth_file_path, map_location=self.device)
        if depth_data.ndim == 3:
            depth_data = depth_data.unsqueeze(1)
        depth_data = torch.nn.functional.interpolate(
            depth_data.float(),
            size=(self.config.resize_height, self.config.resize_width),
            mode="nearest-exact",
        ).squeeze(1)
        self.depth_data = depth_data
        print(f"Loaded depth data from {depth_file_path} with shape: {tuple(depth_data.shape)}")
        return depth_data

    def extract_depth_from_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        if self.depth_data is None:
            raise RuntimeError("Depth data has not been loaded yet. Call extract_depth_from_file first.")
        return self.depth_data
