import argparse
from pathlib import Path
from typing import Optional

import torch
import yaml

from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.planner.direct_shooting import DirectShootingConfig, DirectShootingPlanner
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

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    load_robot_description = None


DEFAULT_FRANKA_DESCRIPTION_NAMES = (
    "franka_panda_description",
    "panda_description",
)
DEFAULT_FRANKA_TARGET_LINK = "panda_grasptarget"


def _build_planner_config(
    scene_dir: Path,
    config_path: Optional[str],
    urdf_path: Optional[str],
    target_link_name: Optional[str],
) -> DirectShootingConfig:
    if config_path:
        with open(Path(config_path).expanduser().resolve(), "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
    else:
        config_data = {}

    if urdf_path:
        config_data["urdf_path"] = urdf_path
    if target_link_name:
        config_data["target_link_name"] = target_link_name

    config_data = _apply_default_franka_config(config_data)

    if "urdf_path" not in config_data and "urdf" not in config_data:
        config_data["urdf_path"] = str(
            ensure_existing_file(
                prompt_scene_path(scene_dir, "Robot URDF path", "robot.urdf", None),
                "Robot URDF",
            )
        )
    if "target_link_name" not in config_data:
        config_data["target_link_name"] = DEFAULT_FRANKA_TARGET_LINK

    return DirectShootingConfig(**config_data)


def _apply_default_franka_config(config_data: dict) -> dict:
    if "target_link_name" not in config_data:
        config_data["target_link_name"] = DEFAULT_FRANKA_TARGET_LINK

    urdf_path = config_data.get("urdf_path")
    if urdf_path and urdf_path != "path/to/robot.urdf":
        return config_data

    if load_robot_description is None:
        config_data.pop("urdf_path", None)
        return config_data

    for description_name in DEFAULT_FRANKA_DESCRIPTION_NAMES:
        try:
            config_data["urdf"] = load_robot_description(description_name)
            config_data.pop("urdf_path", None)
            print(f"Using default Franka robot description: {description_name}")
            return config_data
        except Exception:
            continue

    config_data.pop("urdf_path", None)
    return config_data


def _infer_num_joints(planner: DirectShootingPlanner) -> int:
    candidates = (
        getattr(planner.robot, "num_actuated_joints", None),
        getattr(planner.robot, "num_joints", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate

    joint_limits = getattr(planner.robot, "joint_limits", None)
    if joint_limits is not None and hasattr(joint_limits, "shape") and len(joint_limits.shape) > 0:
        return int(joint_limits.shape[0])

    raise AttributeError(
        "Could not infer the robot joint count from the planner robot. "
        "Pass a robot implementation exposing num_actuated_joints, num_joints, or joint_limits."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a 3D flow result, run direct shooting planning, and visualize in Viser."
    )
    parser.add_argument("--scene-dir")
    parser.add_argument("--flow-result")
    parser.add_argument("--planner-config")
    parser.add_argument("--urdf-path")
    parser.add_argument("--target-link-name")
    parser.add_argument("--initial-joints")
    parser.add_argument("--initial-pose")
    parser.add_argument("--plan-output")
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

    print_banner("Plan And Visualize Flow")
    scene_dir = resolve_scene_dir(args.scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    flow_result_path = ensure_existing_file(
        prompt_scene_path(
            scene_dir,
            "Flow result path",
            "object_flow_result.pt",
            args.flow_result,
        ),
        "Flow result",
    )
    flow_result = ObjectFlowResult.load(str(flow_result_path), device=args.device)

    print_section("Configuration")
    print_kv("Scene Directory", scene_dir)
    print_kv("Flow Result", flow_result_path)
    print_kv("Device", args.device)

    planner_config_path = prompt_scene_path(
        scene_dir,
        "Planner config path",
        "direct_shooting_config.yaml",
        args.planner_config,
    )
    config_path = str(planner_config_path) if planner_config_path.is_file() else None
    print_kv("Planner Config", config_path or "<package defaults>")
    planner_config = _build_planner_config(
        scene_dir,
        config_path,
        args.urdf_path,
        args.target_link_name,
    )
    print_kv("Target Link", planner_config.target_link_name)
    if planner_config.urdf_path:
        print_kv("Robot URDF", planner_config.urdf_path)
    else:
        print_kv("Robot Description", "Default Franka")
    planner = DirectShootingPlanner(planner_config)
    num_joints = _infer_num_joints(planner)

    initial_joints = resolve_float_list(
        args.initial_joints,
        scene_dir,
        "initial_joints.txt",
        num_joints,
        "Initial joints",
    )
    initial_pose = resolve_float_list(
        args.initial_pose,
        scene_dir,
        "initial_pose.txt",
        7,
        "Initial end-effector pose x,y,z,qx,qy,qz,qw",
    )
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
        "Plan output path",
        "direct_shooting_plan.pt",
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
    print_banner("Planner Ready")
    print_kv("Saved Plan", plan_output_path)
    print_kv("Plan Timesteps", plan.joint_trajectory.shape[0])

    viewer = Dream2FlowViewer.get_viewer(port=args.viser_port)
    viewer.visualize_object_flow(
        flow_result.flow,
        name="/scripts/object_flow",
        show_all_timesteps=args.show_all_timesteps,
    )
    viewer.visualize_batched_axes(
        name="/scripts/plan",
        positions=torch.from_numpy(plan.ee_trajectory[:, :3]).float(),
        rotations=torch.from_numpy(plan.ee_trajectory[:, 3:]).float(),
    )

    print_kv("Viser URL", f"http://localhost:{args.viser_port}")
    input("Press Enter to close the viewer...")


if __name__ == "__main__":
    main()
