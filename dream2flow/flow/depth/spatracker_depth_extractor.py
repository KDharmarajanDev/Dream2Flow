from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from dream2flow.flow.depth.base import DepthExtractor, DepthExtractorConfig


@dataclass
class SpaTrackerDepthExtractorConfig(DepthExtractorConfig):
    pass


class SpaTrackerDepthExtractor(DepthExtractor):
    def __init__(self, config: SpaTrackerDepthExtractorConfig, device: str) -> None:
        super().__init__(config, device)

        workspace_root = Path(__file__).resolve().parents[4]
        spatracker_root = workspace_root / "video-particles" / "deps" / "SpaTrackerV2"
        if str(spatracker_root) not in sys.path:
            sys.path.insert(0, str(spatracker_root))

        from models.SpaTrackV2.models.predictor import Predictor
        from models.SpaTrackV2.models.utils import get_points_on_a_grid
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
        from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

        self.vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        self.vggt4track_model.eval()
        self.vggt4track_model = self.vggt4track_model.to(self.device)

        self.tracking_model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
        self.tracking_model.spatrack.track_num = 756
        self.tracking_model.eval()
        self.tracking_model.to(self.device)

        self._preprocess_image_fn = preprocess_image
        self._grid_point_get_fn = get_points_on_a_grid

    def extract_depth_from_file(self, scene_dir: str, video_path: str) -> torch.Tensor:
        resolved_video_path = Path(video_path)
        if not resolved_video_path.is_absolute():
            resolved_video_path = Path(scene_dir) / video_path
        cap = cv2.VideoCapture(str(resolved_video_path))
        frames_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        cap.release()
        frames_np = np.stack(frames_list)
        video_frames = torch.from_numpy(frames_np).to(self.device)
        video_frames = video_frames.permute(0, 3, 1, 2)
        return self.extract_depth_from_video(video_frames)

    def extract_depth_from_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        video_frames = video_frames.float() / 255.0
        video_frames = self._preprocess_image_fn(video_frames)[None]

        autocast_dtype = torch.bfloat16
        with torch.inference_mode():
            if self.device.startswith("cuda"):
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    geometry_predictions = self.vggt4track_model(video_frames)
            else:
                geometry_predictions = self.vggt4track_model(video_frames)
            extrinsic = geometry_predictions["poses_pred"]
            intrinsic = geometry_predictions["intrs"]
            depth_map = geometry_predictions["points_map"][..., 2]
            depth_conf = geometry_predictions["unc_metric"]

        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_frames.squeeze()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

        frame_h, frame_w = video_tensor.shape[2:]
        grid_pts = self._grid_point_get_fn(10, (frame_h, frame_w), device="cpu")
        query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

        if self.device.startswith("cuda"):
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = self.tracking_model.forward(
                    video_tensor,
                    depth=depth_tensor,
                    intrs=intrs,
                    extrs=extrs,
                    queries=query_xyt,
                    fps=1,
                    full_point=False,
                    iters_track=4,
                    query_no_BA=True,
                    fixed_cam=False,
                    stage=1,
                    unc_metric=unc_metric,
                    support_frame=len(video_tensor) - 1,
                    replace_ratio=0.2,
                )
        else:
            outputs = self.tracking_model.forward(
                video_tensor,
                depth=depth_tensor,
                intrs=intrs,
                extrs=extrs,
                queries=query_xyt,
                fps=1,
                full_point=False,
                iters_track=4,
                query_no_BA=True,
                fixed_cam=False,
                stage=1,
                unc_metric=unc_metric,
                support_frame=len(video_tensor) - 1,
                replace_ratio=0.2,
            )

        point_map = outputs[2]
        conf_depth = outputs[3]
        depth_save = point_map[:, 2, ...]
        depth_save[conf_depth < 0.5] = 0
        depth_save = torch.nn.functional.interpolate(
            depth_save.unsqueeze(1),
            size=(self.config.resize_height, self.config.resize_width),
            mode="nearest",
        ).squeeze(1)
        return depth_save
