import numpy as np
import torch
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from dream2flow.planner.base import Planner, PlanResult
from dream2flow.flow.object_flow_result import ObjectFlowResult

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    load_robot_description = None

@dataclass
class TrajectoryOptimizerConfig:
    """Configuration for TrajectoryOptimizer."""
    urdf_path: str = "panda_description"
    target_link_name: str = "panda_hand_tcp"
    path_length_weight: float = 1.0
    particle_matching_weight: float = 50.0
    max_iterations: int = 100
    visualize: bool = False
    max_num_timesteps_for_optimization: int = 50

class TrajectoryOptimizer(Planner):
    """
    Planner that uses PyRoki joint-space optimization to solve for robot trajectories.
    """

    def __init__(self, config: TrajectoryOptimizerConfig):
        self.config = config
        if load_robot_description is not None:
            try:
                self.urdf = load_robot_description(config.urdf_path)
            except Exception:
                self.urdf = yourdfpy.URDF.load(config.urdf_path)
        else:
            self.urdf = yourdfpy.URDF.load(config.urdf_path)
        self.robot = pk.Robot.from_urdf(self.urdf)
        
        if config.target_link_name not in self.robot.links.names:
            raise ValueError(f"Target link {config.target_link_name} not found in URDF")
        self.target_link_index = self.robot.links.names.index(config.target_link_name)

    @property
    def num_actuated_joints(self) -> int:
        return int(self.robot.joints.num_actuated_joints)

    def get_end_effector_pose(self, joints: np.ndarray) -> np.ndarray:
        joint_array = jnp.asarray(joints)
        Ts_world = self.robot.forward_kinematics(joint_array)
        ee_se3 = jaxlie.SE3(Ts_world[self.target_link_index])
        ee_pos = np.asarray(ee_se3.translation())
        ee_quat_wxyz = np.asarray(ee_se3.rotation().wxyz)
        ee_quat_xyzw = np.concatenate([ee_quat_wxyz[1:], ee_quat_wxyz[:1]], axis=-1)
        return np.concatenate([ee_pos, ee_quat_xyzw], axis=-1)

    def plan(self, 
             flow_result: ObjectFlowResult, 
             initial_joints: np.ndarray, 
             initial_pose: np.ndarray,
             **kwargs) -> PlanResult:
        """
        Solve for optimal joint trajectory using PyRoki optimization.
        """
        # Subsample flow if too long
        flow = flow_result.flow
        T_total = flow.position.shape[0]
        step_size = max(1, T_total // self.config.max_num_timesteps_for_optimization)
        
        target_positions = flow.position[::step_size].detach().cpu().numpy()
        target_valid_mask = flow.valid_mask[::step_size].detach().cpu().numpy()
        
        # initial_pose is (7,) [x, y, z, qx, qy, qz, qw]
        # initial_joints is (num_joints,)
        
        num_timesteps = target_positions.shape[0]
        
        # Solve optimization
        joint_trajectory, ee_poses = self._solve_pyroki_optimization(
            initial_joints=initial_joints,
            initial_pose=initial_pose,
            target_particle_positions=jnp.array(target_positions),
            target_valid_mask=jnp.array(target_valid_mask),
            num_timesteps=num_timesteps
        )
        
        return PlanResult(
            joint_trajectory=np.array(joint_trajectory),
            ee_trajectory=np.array(ee_poses)
        )

    def _solve_pyroki_optimization(self,
                                   initial_joints: np.ndarray,
                                   initial_pose: np.ndarray,
                                   target_particle_positions: jnp.ndarray,
                                   target_valid_mask: jnp.ndarray,
                                   num_timesteps: int):
        
        target_links = self.target_link_index
        robot = self.robot

        # Define cost factors
        @jaxls.Cost.create_factory(name="MatchStartJointsCost")
        def match_start_joints_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
            return (vals[joint_var] - initial_joints).flatten() * 100.0

        @jaxls.Cost.create_factory(name="BatchedParticleTrackingCost")
        def batched_particle_tracking_cost(
            vals: jaxls.VarValues,
            robot: pk.Robot,
            joint_var: jaxls.Var[jnp.ndarray],
            initial_particles_local: jnp.ndarray,
            target_particles_world: jnp.ndarray,
            valid_pairs_mask: jnp.ndarray,
            weight: float = 1.0,
        ) -> jnp.ndarray:
            joint_cfgs = vals[joint_var]
            Ts_link_world = robot.forward_kinematics(joint_cfgs)
            ee_se3 = jaxlie.SE3(Ts_link_world[..., target_links, :])
            
            # initial_particles_local: (T, N, 3)
            transformed_positions = ee_se3 @ initial_particles_local
            diffs = transformed_positions - target_particles_world
            distances = jnp.linalg.norm(diffs, axis=-1)
            
            valid_distances = distances * valid_pairs_mask
            valid_count = jnp.sum(valid_pairs_mask, axis=-1)
            mean_distance = jnp.sum(valid_distances, axis=-1) / jnp.maximum(valid_count, 1.0)
            
            return jnp.array([jnp.mean(mean_distance) * weight])

        # Setup variables
        traj_var = robot.joint_var_cls(jnp.arange(num_timesteps))
        traj_var_prev = robot.joint_var_cls(jnp.arange(num_timesteps - 1))
        traj_var_next = robot.joint_var_cls(jnp.arange(1, num_timesteps))

        # Transform initial particles to local EE frame
        initial_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(initial_pose[3:]), 
            initial_pose[:3]
        )
        
        # Get particles at t=0
        initial_particles_world = target_particle_positions[0]
        initial_particles_local_single = initial_se3.inverse() @ initial_particles_world
        initial_particles_local = jnp.broadcast_to(initial_particles_local_single, (num_timesteps, *initial_particles_local_single.shape))

        # Factors
        factors = []
        factors.append(match_start_joints_cost(robot.joint_var_cls(0)))
        
        # Particle tracking weight scaling
        num_total_particles = target_valid_mask.shape[1]
        num_valid_particles = jnp.sum(target_valid_mask)
        scaling_factor = (target_valid_mask.shape[0] * num_total_particles) / jnp.maximum(num_valid_particles, 1.0)
        particle_weight = self.config.particle_matching_weight * scaling_factor
        
        batched_robot = jax.tree.map(lambda x: x[None], robot)
        
        factors.append(
            batched_particle_tracking_cost(
                robot=batched_robot,
                joint_var=traj_var,
                initial_particles_local=initial_particles_local,
                target_particles_world=target_particle_positions,
                valid_pairs_mask=target_valid_mask,
                weight=particle_weight
            )
        )
        
        factors.append(
            pk.costs.smoothness_cost(
                traj_var_prev,
                traj_var_next,
                weight=self.config.path_length_weight * 10.0
            )
        )
        
        factors.append(
            pk.costs.limit_cost(
                batched_robot,
                traj_var,
                weight=100.0
            )
        )

        # Initial guess: repeat initial joints
        initial_trajectory = jnp.tile(initial_joints[None, :], (num_timesteps, 1))
        
        # Solve
        solve_result = (
            jaxls.LeastSquaresProblem(factors, [traj_var])
            .analyze()
            .solve(
                verbose=False,
                initial_vals=jaxls.VarValues.make((traj_var.with_value(initial_trajectory),)),
                termination=jaxls.TerminationConfig(max_iterations=self.config.max_iterations),
                trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
            )
        )
        solution = solve_result[0] if isinstance(solve_result, tuple) else solve_result
        
        joint_trajectory = solution[traj_var]
        
        # Forward kinematics for EE poses
        Ts_world = robot.forward_kinematics(joint_trajectory)
        ee_se3 = jaxlie.SE3(Ts_world[..., target_links, :])
        ee_pos = ee_se3.translation()
        ee_quat_wxyz = ee_se3.rotation().wxyz
        # Convert wxyz to xyzw
        ee_quat_xyzw = jnp.concatenate([ee_quat_wxyz[..., 1:], ee_quat_wxyz[..., 0:1]], axis=-1)
        ee_poses = jnp.concatenate([ee_pos, ee_quat_xyzw], axis=-1)
        
        return joint_trajectory, ee_poses
