import argparse
from pathlib import Path
from typing import Optional

import torch

from dream2flow.camera import CameraCalibration
from dream2flow.flow.generators.video_flow_generator import VideoFlowGenerator
from dream2flow.visualization.viewer import Dream2FlowViewer


def _prompt_if_missing(value: Optional[str], message: str) -> str:
    if value:
        return value
    while True:
        response = input(message).strip()
        if response:
            return response
        print("A value is required.")


def _resolve_output_path(output_path: Optional[str], output_dir: Optional[str]) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()

    directory = Path(output_dir or "outputs").expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "object_flow_result.pt"


def _load_tensor(path: str, *, device: str) -> torch.Tensor:
    tensor = torch.load(Path(path).expanduser().resolve(), map_location=device)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor).__name__}")
    return tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3D object flow result from saved depth maps and 2D tracks."
    )
    parser.add_argument("--camera-calibration", dest="camera_calibration")
    parser.add_argument("--camera-name")
    parser.add_argument("--start-image")
    parser.add_argument("--depth-frames")
    parser.add_argument("--pixel-coords-2d")
    parser.add_argument("--tracks-2d")
    parser.add_argument("--output-path")
    parser.add_argument("--output-dir")
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

    device = args.device
    camera_calibration_path = _prompt_if_missing(
        args.camera_calibration, "Path to camera calibration JSON: "
    )
    camera_name = _prompt_if_missing(args.camera_name, "Camera name: ")
    start_image_path = _prompt_if_missing(
        args.start_image, "Path to start image tensor (.pt): "
    )
    depth_frames_path = _prompt_if_missing(
        args.depth_frames, "Path to depth frames tensor (.pt): "
    )
    pixel_coords_path = _prompt_if_missing(
        args.pixel_coords_2d, "Path to initial pixel coordinates tensor (.pt): "
    )
    tracks_path = _prompt_if_missing(
        args.tracks_2d, "Path to tracked 2D trajectories tensor (.pt): "
    )

    output_path = _resolve_output_path(args.output_path, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera = CameraCalibration.load(camera_calibration_path, device=device)
    start_image = _load_tensor(start_image_path, device=device)
    depth_frames = _load_tensor(depth_frames_path, device=device)
    pixel_coords_2d = _load_tensor(pixel_coords_path, device=device)
    tracks_2d = _load_tensor(tracks_path, device=device)

    generator = VideoFlowGenerator()
    flow_result = generator.generate(
        output_dir="",
        start_image=start_image,
        camera=camera,
        camera_name=camera_name,
        depth_frames=depth_frames,
        pixel_coords_2d=pixel_coords_2d,
        tracks_2d=tracks_2d,
    )
    flow_result.save(str(output_path))

    viewer = Dream2FlowViewer.get_viewer(port=args.viser_port)
    camera.visualize(viewer)
    viewer.visualize_object_flow(
        flow_result.flow,
        name="/examples/object_flow",
        show_all_timesteps=args.show_all_timesteps,
    )

    print(f"Saved 3D flow to {output_path}")
    print(f"Viser viewer running at http://localhost:{args.viser_port}")
    input("Press Enter to close the viewer...")


if __name__ == "__main__":
    main()
