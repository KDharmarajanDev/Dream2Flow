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

### Scripts

The package includes two runnable scripts under `dream2flow/scripts/`. Both are organized around a scene directory: one directory per scene, containing the files for that scene. Each prompt suggests a default filename in parentheses, interpreted relative to the chosen scene directory. You can press Enter to accept the default, or provide a full path to override it.

#### `dream2flow.scripts.create_3d_flow`

Run with:

```bash
python -m dream2flow.scripts.create_3d_flow
```

Pipeline choices:

- Video generation method:
  `[1] local file`
  `[2] Veo 3`
- Depth estimation mode:
  `[1] playback`
  `[2] generate`

Scene file defaults:

- Camera calibration: `camera_calibration_info.json`
- Start RGB image: `camera_rgb.png`
- Language instruction: `language_instruction.txt`
- Local video file: `rgb.mp4`
- Playback depth frames: `depth_frames.pt`
- Initial depth for generated depth: `initial_depth.pt`
- Saved 2D tracks: `tracks_2d.pt`
- Output 3D flow result: `object_flow_result.pt`

Input logic:

- The script always reads the language instruction from `language_instruction.txt`
- If video generation uses `local file`, it reads a video file, defaulting to `rgb.mp4`
- If video generation uses `Veo 3`, it uses the start image and language instruction to generate a new video in the scene directory
- If depth estimation uses `playback`, it reads `depth_frames.pt`
- If depth estimation uses `generate`, it reads `initial_depth.pt` and writes `depth_frames.pt`
- After video generation and depth preparation, the script runs CoTrackerV3 offline tracking, saves `tracks_2d.pt`, and lifts the tracks into 3D

Outputs:

- Saves an `ObjectFlowResult` `.pt` file, by default at `<scene_dir>/object_flow_result.pt`
- Saves `tracks_2d.pt` in the scene directory
- Opens a Viser session, by default at `http://localhost:8080`

Required file formats:

- Camera calibration JSON:
  created by `CameraCalibration.save(...)`
- Camera calibration JSON structure:
  top-level mapping keyed by camera name
- Camera entry fields:
  `intrinsics`: 3x3 numeric matrix
  `extrinsics`: 4x4 numeric matrix
- Start RGB image:
  RGB `.png` image
- Language instruction file:
  plain text file containing a single sentence
- Local video file:
  `.mp4` video readable by OpenCV
- Depth frames tensor `.pt`:
  `torch.Tensor` with shape `(T, H, W)`
- Initial depth tensor `.pt`:
  `torch.Tensor` with shape `(H, W)` or `(1, H, W)`

#### `dream2flow.scripts.plan_and_visualize_flow`

This script requires the planner dependencies:

```bash
python -m pip install -e ".[planner]"
```

Run with:

```bash
python -m dream2flow.scripts.plan_and_visualize_flow
```

Scene file defaults:

- flow result: `object_flow_result.pt`
- planner config: `direct_shooting_config.yaml`
- initial joints: `initial_joints.txt`
- initial pose: `initial_pose.txt`
- plan output: `direct_shooting_plan.pt`

Input logic:

- The script first looks for the flow result in the scene directory, unless you override it with a full path
- The script looks for `initial_joints.txt` and `initial_pose.txt` in the scene directory before prompting for inline values
- If `direct_shooting_config.yaml` is not present in the scene directory, the script falls back to the packaged defaults and then applies any explicit overrides

Default robot behavior:

- The script first tries to load a Franka robot through `robot_descriptions.loaders.yourdfpy.load_robot_description(...)`
- The default target link is `panda_grasptarget`
- If the Franka description cannot be loaded and no URDF path is provided, the script falls back to a scene-local URDF path with the default filename `robot.urdf`

Required file formats:

- Flow result `.pt`:
  a serialized `ObjectFlowResult` saved by Dream2Flow, typically from `dream2flow.scripts.create_3d_flow`
- Planner YAML config:
  YAML mapping compatible with `DirectShootingConfig`
- Planner YAML supported keys:
  `urdf_path`: string or null
  `target_link_name`: string
  `path_length_weight`: float
  `particle_matching_weight`: float
  `max_iterations`: integer
  `visualize`: boolean
  `ee2tip_offset`: list of 3 numbers
  `max_num_timesteps_for_optimization`: integer
- Initial joints text file:
  comma-separated joint values, one scene-specific robot configuration
- Initial pose text file:
  comma-separated `x,y,z,qx,qy,qz,qw`

Outputs:

- Saves a planner result `.pt` file containing:
  `joint_trajectory`: tensor with shape `(T, J)`
  `ee_trajectory`: tensor with shape `(T, 7)`
- Default output path:
  `<scene_dir>/direct_shooting_plan.pt`
- Opens a Viser session showing both the object flow and the planned trajectory

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
