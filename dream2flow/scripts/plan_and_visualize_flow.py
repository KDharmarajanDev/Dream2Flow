import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from viser.extras import ViserUrdf

from dream2flow.camera import CameraCalibration
from dream2flow.flow.geometry import depth_to_world_points
from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.planner.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerConfig
from dream2flow.scripts._scene_utils import (
    ensure_existing_file,
    print_banner,
    print_kv,
    print_section,
    prompt_scene_path,
    resolve_float_list,
    resolve_scene_dir,
)
from dream2flow.visualization.viewer import Dream2FlowViewer

DEFAULT_URDF_PATH = "panda_description"
DEFAULT_TARGET_LINK = "panda_hand_tcp"
DEFAULT_FRANKA_ARM_JOINTS = [0.0] * 7
DEFAULT_FRANKA_ENV_PATH = (
    Path(__file__).resolve().parents[3] / "svl-franka-tutorial" / "franka_env.py"
)


@dataclass(frozen=True)
class SceneDataConfig:
    instruction: str
    object_name: str
    robot_start_joints: tuple[float, ...] | None = None


def _build_planner_config(
    scene_dir: Path,
    config_path: Optional[str],
    urdf_path: Optional[str],
    target_link_name: Optional[str],
) -> TrajectoryOptimizerConfig:
    if config_path:
        with open(Path(config_path).expanduser().resolve(), "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
    else:
        config_data = {}

    if urdf_path:
        config_data["urdf_path"] = urdf_path
    if target_link_name:
        config_data["target_link_name"] = target_link_name
    if "urdf_path" not in config_data:
        config_data["urdf_path"] = DEFAULT_URDF_PATH
    if "target_link_name" not in config_data:
        config_data["target_link_name"] = DEFAULT_TARGET_LINK

    return TrajectoryOptimizerConfig(**config_data)


def _extract_reset_joint_positions(franka_env_path: Path) -> list[float]:
    module = ast.parse(franka_env_path.read_text(encoding="utf-8"))

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "RESET_JOINT_POSITIONS":
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "np"
                    and value.func.attr == "array"
                    and value.args
                ):
                    return [float(x) for x in ast.literal_eval(value.args[0])]

    raise ValueError(f"Could not find RESET_JOINT_POSITIONS in {franka_env_path}")


def _default_initial_joints(num_joints: int) -> list[float]:
    try:
        joint_values = _extract_reset_joint_positions(DEFAULT_FRANKA_ENV_PATH)
    except Exception:
        joint_values = list(DEFAULT_FRANKA_ARM_JOINTS)

    if num_joints > len(joint_values):
        joint_values.extend([0.0] * (num_joints - len(joint_values)))
    return joint_values[:num_joints]


def _load_scene_data_config(path: Path) -> SceneDataConfig:
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

    robot_start_joints = data.get("robot_start_joints")
    if robot_start_joints is not None:
        robot_start_joints = tuple(float(value) for value in robot_start_joints)

    return SceneDataConfig(
        instruction=instruction,
        object_name=object_name,
        robot_start_joints=robot_start_joints,
    )


def _plan_timestep_indices(num_flow_timesteps: int, max_num_timesteps: int) -> np.ndarray:
    step_size = max(1, num_flow_timesteps // max_num_timesteps)
    return np.arange(0, num_flow_timesteps, step_size, dtype=np.int64)


def _xyzw_to_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return np.array(
        [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ],
        dtype=np.float64,
    )


def _subsample_point_cloud(
    points: torch.Tensor,
    colors: torch.Tensor,
    max_points: int = 50000,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] <= max_points:
        return points, colors
    indices = torch.linspace(0, points.shape[0] - 1, max_points, device=points.device).long()
    return points[indices], colors[indices]


def _load_tensor(path: str, *, device: str) -> torch.Tensor:
    tensor = torch.load(Path(path).expanduser().resolve(), map_location=device)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor).__name__}")
    return tensor


def _load_rgb_image(path: str, *, device: str) -> torch.Tensor:
    image = Image.open(Path(path).expanduser().resolve()).convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return image_tensor.to(device)


def _suggest_camera_name(camera_calibration_path: Path) -> str:
    with open(camera_calibration_path, "r", encoding="utf-8") as file:
        calibration_data = json.load(file)
    if len(calibration_data) == 1:
        return next(iter(calibration_data))
    return "front_camera"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a 3D flow result, run trajectory optimization, and visualize in Viser."
    )
    parser.add_argument("--scene-dir")
    parser.add_argument("--flow-result")
    parser.add_argument("--planner-config")
    parser.add_argument("--urdf-path")
    parser.add_argument("--target-link-name")
    parser.add_argument("--scene-data")
    parser.add_argument("--camera-calibration", dest="camera_calibration")
    parser.add_argument("--camera-name")
    parser.add_argument("--start-image")
    parser.add_argument("--initial-depth")
    parser.add_argument("--initial-joints")
    parser.add_argument("--initial-pose")
    parser.add_argument("--plan-output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viser-port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print_banner("Plan And Visualize Flow")
    scene_dir = resolve_scene_dir(args.scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    flow_result_path = ensure_existing_file(
        prompt_scene_path(
            scene_dir,
            "Flow result path",
            "object_flow_result.pt",
            args.flow_result,
            prompt_if_missing=True,
        ),
        "Flow result",
    )
    flow_result = ObjectFlowResult.load(str(flow_result_path), device=args.device)

    print_section("Configuration")
    print_kv("Scene Directory", scene_dir)
    print_kv("Flow Result", flow_result_path)
    print_kv("Device", args.device)

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
    camera_name = args.camera_name or _suggest_camera_name(camera_calibration_path)
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
    scene_data_path = ensure_existing_file(
        prompt_scene_path(
            scene_dir,
            "Scene data",
            "scene_data.yaml",
            args.scene_data,
            prompt_if_missing=True,
        ),
        "Scene data",
    )
    scene_data_config = _load_scene_data_config(scene_data_path)

    planner_config_path = prompt_scene_path(
        scene_dir,
        "Trajectory optimizer config path",
        "trajectory_optimization_config.yaml",
        args.planner_config,
    )
    config_path = str(planner_config_path) if planner_config_path.is_file() else None
    print_kv("Trajectory Optimizer Config", config_path or "<package defaults>")
    print_kv("Scene Data", scene_data_path)
    print_kv("Camera Calibration", camera_calibration_path)
    print_kv("Camera Name", camera_name)
    print_kv("Start RGB Image", start_image_path)
    print_kv("Initial Depth", initial_depth_path)
    planner_config = _build_planner_config(
        scene_dir,
        config_path,
        args.urdf_path,
        args.target_link_name,
    )
    print_kv("Target Link", planner_config.target_link_name)
    print_kv("Robot URDF", planner_config.urdf_path)
    planner = TrajectoryOptimizer(planner_config)
    num_joints = planner.num_actuated_joints

    default_joint_values = scene_data_config.robot_start_joints or tuple(_default_initial_joints(num_joints))
    initial_joints = resolve_float_list(
        args.initial_joints or ",".join(str(value) for value in default_joint_values),
        scene_dir,
        "initial_joints.txt",
        num_joints,
        "Initial joints",
    )
    initial_pose_path = scene_dir / "initial_pose.txt"
    if args.initial_pose or initial_pose_path.is_file():
        initial_pose = resolve_float_list(
            args.initial_pose,
            scene_dir,
            "initial_pose.txt",
            7,
            "Initial end-effector pose x,y,z,qx,qy,qz,qw",
        )
    else:
        initial_pose = np.asarray(planner.get_end_effector_pose(initial_joints), dtype=np.float64)
    print_kv("Num Joints", num_joints)
    print_kv("Initial Joints", ",".join(str(x) for x in initial_joints))
    print_kv("Initial Pose", ",".join(str(x) for x in initial_pose))

    plan = planner.plan(
        flow_result=flow_result,
        initial_joints=initial_joints,
        initial_pose=initial_pose,
    )

    plan_output_path = prompt_scene_path(
        scene_dir,
        "Trajectory optimization output path",
        "trajectory_optimization_plan.pt",
        args.plan_output,
    )
    plan_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "joint_trajectory": torch.from_numpy(plan.joint_trajectory),
            "ee_trajectory": torch.from_numpy(plan.ee_trajectory),
        },
        plan_output_path,
    )
    print_banner("Trajectory Optimizer Ready")
    print_kv("Saved Trajectory", plan_output_path)
    print_kv("Trajectory Timesteps", plan.joint_trajectory.shape[0])
    flow_timestep_indices = _plan_timestep_indices(
        flow_result.flow.position.shape[0],
        planner_config.max_num_timesteps_for_optimization,
    )
    camera = CameraCalibration.load(str(camera_calibration_path), device=args.device)
    start_image = _load_rgb_image(str(start_image_path), device=args.device)
    initial_depth = _load_tensor(str(initial_depth_path), device=args.device)

    viewer = Dream2FlowViewer.get_viewer(port=args.viser_port)
    _visualize_initial_rgbd_point_cloud(
        viewer,
        camera,
        camera_name,
        start_image,
        initial_depth,
    )
    viewer.visualize_flow_timestep(
        flow_result.flow,
        name="/scripts/object_flow/current",
        timestep=int(flow_timestep_indices[0]),
        point_size=0.007,
    )

    current_ee_pose = np.asarray(plan.ee_trajectory[0], dtype=np.float64)
    current_ee_handle = viewer.server.scene.add_frame(
        "/scripts/plan/current_ee",
        position=current_ee_pose[:3],
        wxyz=_xyzw_to_wxyz(current_ee_pose[3:]),
        show_axes=True,
        axes_length=0.08,
        axes_radius=0.008,
    )
    robot_vis = ViserUrdf(
        viewer.server,
        planner.urdf,
        root_node_name="/scripts/robot",
    )

    timestep_slider = viewer.server.gui.add_slider(
        "timestep",
        min=0,
        max=plan.joint_trajectory.shape[0] - 1,
        step=1,
        initial_value=0,
    )

    def _update_visualization(timestep: int) -> None:
        plan_timestep = int(np.clip(timestep, 0, plan.joint_trajectory.shape[0] - 1))
        flow_timestep = int(flow_timestep_indices[plan_timestep])
        ee_pose = np.asarray(plan.ee_trajectory[plan_timestep], dtype=np.float64)

        with viewer.server.atomic():
            viewer.visualize_flow_timestep(
                flow_result.flow,
                name="/scripts/object_flow/current",
                timestep=flow_timestep,
                point_size=0.007,
            )
            current_ee_handle.position = ee_pose[:3]
            current_ee_handle.wxyz = _xyzw_to_wxyz(ee_pose[3:])
            robot_vis.update_cfg(np.asarray(plan.joint_trajectory[plan_timestep], dtype=np.float64))

    @timestep_slider.on_update
    def _(_) -> None:
        _update_visualization(timestep_slider.value)

    _update_visualization(0)

    print_kv("Viser URL", f"http://localhost:{args.viser_port}")
    try:
        input("Press Enter to close the viewer...")
    except EOFError:
        pass


if __name__ == "__main__":
    main()
