import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.planner.direct_shooting import DirectShootingConfig, DirectShootingPlanner
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


def _prompt_if_missing(value: Optional[str], message: str) -> str:
    if value:
        return value
    while True:
        response = input(message).strip()
        if response:
            return response
        print("A value is required.")


def _parse_float_list(
    value: Optional[str],
    expected_length: int,
    prompt_message: str,
    *,
    dtype=np.float64,
) -> np.ndarray:
    while True:
        raw_value = value if value is not None else input(prompt_message).strip()
        try:
            parsed = np.fromstring(raw_value, sep=",", dtype=dtype)
        except ValueError:
            parsed = np.array([], dtype=dtype)

        if parsed.size == expected_length:
            return parsed

        print(f"Expected {expected_length} comma-separated values.")
        value = None


def _build_planner_config(
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
        config_data["urdf_path"] = _prompt_if_missing(None, "Path to robot URDF: ")
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

    flow_result_path = _prompt_if_missing(
        args.flow_result, "Path to object flow result (.pt): "
    )
    flow_result = ObjectFlowResult.load(flow_result_path, device=args.device)

    planner_config = _build_planner_config(
        args.planner_config, args.urdf_path, args.target_link_name
    )
    planner = DirectShootingPlanner(planner_config)
    num_joints = _infer_num_joints(planner)

    initial_joints = _parse_float_list(
        args.initial_joints,
        expected_length=num_joints,
        prompt_message=(f"Initial joints ({num_joints} comma-separated values): "),
    )
    initial_pose = _parse_float_list(
        args.initial_pose,
        expected_length=7,
        prompt_message="Initial end-effector pose x,y,z,qx,qy,qz,qw: ",
    )

    plan = planner.plan(
        flow_result=flow_result,
        initial_joints=initial_joints,
        initial_pose=initial_pose,
    )

    if args.plan_output:
        plan_output_path = Path(args.plan_output).expanduser().resolve()
        plan_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "joint_trajectory": torch.from_numpy(plan.joint_trajectory),
                "ee_trajectory": torch.from_numpy(plan.ee_trajectory),
            },
            plan_output_path,
        )
        print(f"Saved planner output to {plan_output_path}")

    viewer = Dream2FlowViewer.get_viewer(port=args.viser_port)
    viewer.visualize_object_flow(
        flow_result.flow,
        name="/examples/object_flow",
        show_all_timesteps=args.show_all_timesteps,
    )
    viewer.visualize_batched_axes(
        name="/examples/plan",
        positions=torch.from_numpy(plan.ee_trajectory[:, :3]).float(),
        rotations=torch.from_numpy(plan.ee_trajectory[:, 3:]).float(),
    )

    print(f"Planner trajectory has {plan.joint_trajectory.shape[0]} timesteps.")
    print(f"Viser viewer running at http://localhost:{args.viser_port}")
    input("Press Enter to close the viewer...")


if __name__ == "__main__":
    main()
