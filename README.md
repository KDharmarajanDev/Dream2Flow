# Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow

Dream2Flow is a library for generating 3D object flow from video sources and planning robot trajectories to match that flow.

## Features

- **3D Object Flow**: Clean abstractions for representing and visualizing 3D trajectories of objects.
- **Video Sources**: Support for generating videos via Google Veo 3 or playing back from files.
- **Motion Planning**: Direct Shooting Planner using PyRoki for joint-space optimization.
- **Grasp Planning**: Configurable grasp waypoint sequences relative to object centroids.
- **Visualization**: Interactive 3D visualization using Viser.

## Installation

```bash
pip install -e .
```

For the motion planner, you'll need the following extra dependencies:
```bash
pip install pyroki jax jaxlie jaxls yourdfpy
```

## Usage

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
    urdf_path="assets/panda.urdf",
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
viewer.visualize_batched_axes("robot_plan", positions=plan.ee_trajectory[:, :3], rotations=plan.ee_trajectory[:, 3:])
```

## Citation

```bibtex
@article{dharmarajan2024dream2flow,
  title={Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow},
  author={Dharmarajan, Karthik and Huang, Wenlong and Wu, Jiajun and Fei-Fei, Li and Zhang, Ruohan},
  journal={arXiv preprint arXiv:2403.12345},
  year={2024}
}
```
