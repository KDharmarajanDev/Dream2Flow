# Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow

Official code release for **Dream2Flow**.

[[Website](https://dream2flow.github.io/)] [[arXiv](https://arxiv.org/abs/2512.24766)] [[Paper](https://dream2flow.github.io/paper.pdf)]

[**<u>Karthik Dharmarajan</u>**](https://kdharmarajan.com/), [**<u>Wenlong Huang</u>**](https://huangwl18.github.io/)\*, [**<u>Jiajun Wu</u>**](https://jiajunwu.com/), [**<u>Li Fei-Fei</u>**](https://profiles.stanford.edu/fei-fei-li)†, [**<u>Ruohan Zhang</u>**](https://ai.stanford.edu/~zharu/)†
<br>
\* Corresponding author, † Equal advising
<br>
Stanford University

---

Dream2Flow is a library for generating 3D object flow from video sources and planning robot trajectories to match that flow.

## Features

- **3D Object Flow**: Clean abstractions for representing and visualizing 3D trajectories of objects.
- **Video Sources**: Support for generating videos via Google Veo 3 or playing back from files.
- **Motion Planning**: Direct Shooting Planner using PyRoki for joint-space optimization.
- **Visualization**: Interactive 3D visualization using Viser.

## Installation

Create and activate a conda environment named `dream2flow`:

```bash
conda create -n dream2flow python=3.10 pip -y
conda activate dream2flow
```

Install the base package:

```bash
python -m pip install -e .
```

To include the motion planning stack, install the optional `planner` extra:

```bash
python -m pip install -e ".[planner]"
```

## Usage

### Example Scripts

The package now includes two runnable examples under `dream2flow/examples/`:

- `dream2flow.examples.create_3d_flow` lifts saved 2D tracks and depth maps into a saved `ObjectFlowResult` and opens a Viser viewer.
- `dream2flow.examples.plan_and_visualize_flow` loads a saved `ObjectFlowResult`, runs the direct shooting planner, and visualizes both the flow and planned end-effector trajectory in Viser.

Both scripts accept CLI arguments, and if a required argument is missing they will prompt for it with `input(...)`.

#### 1. Create 3D Flow From Saved Inputs

Prepare the following files first:

- Camera calibration JSON created with `CameraCalibration.save(...)`
- Start image tensor as a `.pt` file with shape `(H, W, 3)`
- Depth frames tensor as a `.pt` file with shape `(T, H, W)`
- Initial pixel coordinates tensor as a `.pt` file with shape `(N, 2)`
- Tracked 2D trajectories tensor as a `.pt` file with shape `(T, N, 2)`

Run with arguments:

```bash
python -m dream2flow.examples.create_3d_flow \
  --camera-calibration path/to/camera_calibration.json \
  --camera-name front_camera \
  --start-image path/to/start_image.pt \
  --depth-frames path/to/depth_frames.pt \
  --pixel-coords-2d path/to/pixel_coords_2d.pt \
  --tracks-2d path/to/tracks_2d.pt \
  --output-path outputs/object_flow_result.pt
```

Or run it with no arguments and provide them interactively:

```bash
python -m dream2flow.examples.create_3d_flow
```

This script saves an `ObjectFlowResult` and opens Viser at `http://localhost:8080` by default.

#### 2. Load Flow, Visualize, And Run The Planner

This example requires the planner dependencies:

```bash
python -m pip install -e ".[planner]"
```

You will need:

- A saved flow result `.pt` file created by the first example
- Initial joints as comma-separated values
- Initial end-effector pose as `x,y,z,qx,qy,qz,qw`

By default, the planner example tries to load a Franka robot from `robot_descriptions.loaders.yourdfpy.load_robot_description(...)` and uses `panda_grasptarget` as the target link. You can still override this with `--urdf-path` and `--target-link-name`, or by passing a planner YAML config such as `dream2flow/configs/direct_shooting_default.yaml`.

```bash
python -m dream2flow.examples.plan_and_visualize_flow \
  --flow-result outputs/object_flow_result.pt \
  --planner-config dream2flow/configs/direct_shooting_default.yaml \
  --initial-joints 0.0,-0.8,0.0,-2.3,0.0,1.6,0.8 \
  --initial-pose 0.4,0.0,0.3,0.0,0.0,0.0,1.0 \
  --plan-output outputs/direct_shooting_plan.pt
```

Or run it interactively:

```bash
python -m dream2flow.examples.plan_and_visualize_flow
```

The planner example visualizes the 3D flow and the planned end-effector frames together in Viser.

### 1. Generate Object Flow from Video

```python
from dream2flow.video.veo import VeoVideoSource
from dream2flow.flow.generators.video_flow_generator import VideoFlowGenerator
from dream2flow.camera import CameraCalibration

# Setup video source
video_source = VeoVideoSource()

# Generate video
video = video_source.generate_video(
    output_dir="outputs/",
    text_prompt="The lamp is moved to the left.",
    start_image=start_image_tensor
)

# Generate 3D flow (requires depth and 2D tracks)
generator = VideoFlowGenerator()
flow_result = generator.generate(
    output_dir="outputs/",
    start_image=start_image_tensor,
    camera=camera_calib,
    camera_name="front_camera",
    depth_frames=depth_frames,
    pixel_coords_2d=initial_pixels,
    tracks_2d=tracks_2d
)
```

### 2. Plan Robot Trajectory

```python
from dream2flow.planner.direct_shooting import DirectShootingPlanner, DirectShootingConfig

config = DirectShootingConfig(
    target_link_name="panda_grasptarget"
)
planner = DirectShootingPlanner(config)

plan = planner.plan(
    flow_result=flow_result,
    initial_joints=current_joints,
    initial_pose=current_ee_pose
)

# plan.joint_trajectory contains the optimized joint sequence
```

### 3. Visualize

```python
from dream2flow.visualization.viewer import Dream2FlowViewer

viewer = Dream2FlowViewer.get_viewer()
viewer.visualize_object_flow(flow_result.flow, name="lamp_flow")
viewer.visualize_batched_axes(
    "robot_plan",
    positions=plan.ee_trajectory[:, :3],
    rotations=plan.ee_trajectory[:, 3:],
)
```

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{dharmarajan2025dream2flow,
  title={Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow},
  author={Dharmarajan, Karthik and Huang, Wenlong and Wu, Jiajun and Fei-Fei, Li and Zhang, Ruohan},
  journal={arXiv preprint arXiv:2512.24766},
  year={2025}
}
```
