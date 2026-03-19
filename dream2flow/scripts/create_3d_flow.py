import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from dream2flow.camera import CameraCalibration
from dream2flow.flow.depth import (
    FileDepthExtractor,
    FileDepthExtractorConfig,
    SpaTrackerDepthExtractor,
    SpaTrackerDepthExtractorConfig,
)
from dream2flow.flow.geometry import depth_to_world_points
from dream2flow.flow.generators.video_flow_generator import VideoFlowGenerator
from dream2flow.scripts._scene_utils import (
    choose_option,
    ensure_existing_file,
    print_banner,
    print_kv,
    print_section,
    prompt_scene_path,
    prompt_with_default,
    resolve_scene_dir,
)
from dream2flow.video.playback import PlaybackVideoSource
from dream2flow.video.veo import VeoVideoSource
from dream2flow.visualization.viewer import Dream2FlowViewer


@dataclass(frozen=True)
class LanguageInstructionConfig:
    instruction: str
    object_name: str


def _subsample_point_cloud(
    points: torch.Tensor,
    colors: torch.Tensor,
    max_points: int = 50000,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] <= max_points:
        return points, colors
    indices = torch.linspace(0, points.shape[0] - 1, max_points, device=points.device).long()
    return points[indices], colors[indices]


def _visualize_initial_rgbd_point_cloud(
    viewer: Dream2FlowViewer,
    camera: CameraCalibration,
    camera_name: str,
    start_image: torch.Tensor,
    initial_depth: torch.Tensor,
) -> None:
    intrinsics, extrinsics = camera.get_camera_calibration(camera_name)
    intrinsics = intrinsics.to(device=initial_depth.device, dtype=torch.float32)
    extrinsics = extrinsics.to(device=initial_depth.device, dtype=torch.float32)
    point_cloud = depth_to_world_points(
        depth=initial_depth.unsqueeze(0).to(dtype=torch.float32),
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
    ).squeeze(0)
    colors = start_image.reshape(-1, 3)
    valid_mask = initial_depth.reshape(-1) > 0
    point_cloud = point_cloud[valid_mask]
    colors = colors[valid_mask]
    if point_cloud.numel() == 0:
        return
    point_cloud, colors = _subsample_point_cloud(point_cloud, colors)
    viewer.visualize_point_cloud(
        "/scripts/initial_rgbd_point_cloud",
        point_cloud,
        colors,
        point_size=0.005,
    )


def _load_tensor(path: str, *, device: str) -> torch.Tensor:
    tensor = torch.load(Path(path).expanduser().resolve(), map_location=device)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor).__name__}")
    return tensor


def _load_rgb_image(path: str, *, device: str) -> torch.Tensor:
    image = Image.open(Path(path).expanduser().resolve()).convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return image_tensor.to(device)


def _load_language_instruction_config(path: Path) -> LanguageInstructionConfig:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")

    instruction = str(data.get("instruction", "")).strip()
    object_name = str(data.get("object_name", "")).strip()
    if not instruction:
        raise ValueError(f"Missing or empty 'instruction' in {path}")
    if not object_name:
        raise ValueError(f"Missing or empty 'object_name' in {path}")

    return LanguageInstructionConfig(instruction=instruction, object_name=object_name)


def _suggest_camera_name(camera_calibration_path: Path) -> str:
    with open(camera_calibration_path, "r", encoding="utf-8") as file:
        calibration_data = json.load(file)
    if len(calibration_data) == 1:
        return next(iter(calibration_data))
    return "front_camera"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D object flow from a scene directory using video, depth, and CoTrackerV3."
    )
    parser.add_argument("--scene-dir")
    parser.add_argument("--video-generation-method")
    parser.add_argument("--video-path")
    parser.add_argument("--depth-mode")
    parser.add_argument("--initial-depth")
    parser.add_argument("--depth-frames")
    parser.add_argument("--camera-calibration", dest="camera_calibration")
    parser.add_argument("--camera-name")
    parser.add_argument("--start-image")
    parser.add_argument("--language-instruction")
    parser.add_argument("--output-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument(
        "--show-all-timesteps",
        action="store_true",
        help="Visualize all flow points at once instead of using a timestep slider.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print_banner("Create 3D Flow")
    scene_dir = resolve_scene_dir(args.scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    camera_calibration_path = ensure_existing_file(
        prompt_scene_path(
            scene_dir,
            "Camera calibration JSON",
            "camera_calibration_info.json",
            args.camera_calibration,
            prompt_if_missing=True,
        ),
        "Camera calibration JSON",
    )
    if args.camera_name:
        camera_name = args.camera_name
    else:
        camera_name = prompt_with_default("Camera name", _suggest_camera_name(camera_calibration_path))

    start_image_path = ensure_existing_file(
        prompt_scene_path(
            scene_dir,
            "Start RGB image",
            "camera_rgb.png",
            args.start_image,
            prompt_if_missing=True,
        ),
        "Start RGB image",
    )
    language_instruction_path = ensure_existing_file(
        prompt_scene_path(
            scene_dir,
            "Language instruction",
            "language_instruction.yaml",
            args.language_instruction,
            prompt_if_missing=True,
        ),
        "Language instruction",
    )

    video_generation_method = choose_option(
        "Video generation method",
        ("local file", "veo 3"),
        "local file",
        args.video_generation_method,
    )
    depth_mode = choose_option(
        "Depth estimation mode",
        ("playback", "generate"),
        "playback",
        args.depth_mode,
    )

    output_path = prompt_scene_path(
        scene_dir,
        "Output flow result path",
        "object_flow_result.pt",
        args.output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_section("Configuration")
    print_kv("Scene Directory", scene_dir)
    print_kv("Device", device)
    print_kv("Camera Calibration", camera_calibration_path)
    print_kv("Camera Name", camera_name)
    print_kv("Start Image", start_image_path)
    print_kv("Language Instruction", language_instruction_path)
    print_kv("Video Generation Method", video_generation_method)
    print_kv("Depth Estimation Mode", depth_mode)
    print_kv("Output Flow Result", output_path)

    camera = CameraCalibration.load(str(camera_calibration_path), device=device)
    start_image = _load_rgb_image(str(start_image_path), device=device)
    language_instruction_config = _load_language_instruction_config(language_instruction_path)
    language_instruction = language_instruction_config.instruction
    object_name = language_instruction_config.object_name
    initial_depth_path = ensure_existing_file(
        prompt_scene_path(
            scene_dir,
            "Initial depth tensor",
            "initial_depth.pt",
            args.initial_depth,
            prompt_if_missing=True,
        ),
        "Initial depth tensor",
    )
    print_kv("Initial Depth", initial_depth_path)
    print_kv("Object Name", object_name)
    initial_depth = _load_tensor(str(initial_depth_path), device=device)

    if video_generation_method == "local file":
        video_path = ensure_existing_file(
            prompt_scene_path(
                scene_dir,
                "Video path",
                "rgb.mp4",
                args.video_path,
                prompt_if_missing=True,
            ),
            "Video path",
        )
        print_kv("Video Path", video_path)
        video_source = PlaybackVideoSource(str(video_path), device=device)
    else:
        print_kv("Video Path", scene_dir / "rgb.mp4")
        video_source = VeoVideoSource(device=device)

    video_frames = video_source.generate_video(
        output_dir=str(scene_dir),
        text_prompt=language_instruction,
        start_image=start_image,
    )

    if depth_mode == "playback":
        depth_frames_path = ensure_existing_file(
            prompt_scene_path(
                scene_dir,
                "Depth frames tensor",
                "depth_frames.pt",
                args.depth_frames,
                prompt_if_missing=True,
            ),
            "Depth frames tensor",
        )
        print_kv("Depth Frames", depth_frames_path)
        depth_extractor = FileDepthExtractor(
            FileDepthExtractorConfig(depth_file_path=str(depth_frames_path)),
            device=device,
        )
    else:
        depth_extractor = SpaTrackerDepthExtractor(SpaTrackerDepthExtractorConfig(), device=device)

    generator = VideoFlowGenerator(depth_extractor=depth_extractor, device=device)
    flow_result = generator.generate(
        output_dir=str(scene_dir),
        start_image=start_image,
        camera=camera,
        camera_name=camera_name,
        instruction=language_instruction,
        object_name=object_name,
        video_frames=video_frames,
        initial_depth=initial_depth,
        video_path=str(video_path) if video_generation_method == "local file" else str(scene_dir / "rgb.mp4"),
    )
    flow_result.save(str(output_path))

    viewer = Dream2FlowViewer.get_viewer(port=args.viser_port)
    camera.visualize(viewer)
    _visualize_initial_rgbd_point_cloud(
        viewer=viewer,
        camera=camera,
        camera_name=camera_name,
        start_image=start_image,
        initial_depth=initial_depth,
    )
    viewer.visualize_object_flow(
        flow_result.flow,
        name="/scripts/object_flow",
        show_all_timesteps=args.show_all_timesteps,
    )

    print_banner("Flow Ready")
    print_kv("Saved 3D Flow", output_path)
    print_kv("Viser URL", f"http://localhost:{args.viser_port}")
    input("Press Enter to close the viewer...")


if __name__ == "__main__":
    main()
