import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.spatial.transform import Rotation as R

@dataclass
class GraspWaypoint:
    """
    A single waypoint in a grasp plan.
    """
    # Position relative to object centroid (3,)
    position: np.ndarray
    
    # Orientation as xyzw quaternion (4,)
    orientation_quat: np.ndarray
    
    # Gripper value (0.0 for open, 1.0 for closed)
    gripper: float

@dataclass
class GraspPlan:
    """
    A sequence of waypoints for a grasp.
    """
    waypoints: List[GraspWaypoint]

    @classmethod
    def from_yaml(cls, filepath: str) -> 'GraspPlan':
        """
        Load grasp plan from a YAML file.
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        waypoints = []
        for wp in data['waypoints']:
            waypoints.append(GraspWaypoint(
                position=np.array(wp['position']),
                orientation_quat=np.array(wp['orientation_quat']),
                gripper=float(wp['gripper'])
            ))
        return cls(waypoints=waypoints)

class ConfigurableGraspPlanner:
    """
    Planner that applies pre-configured grasp waypoints relative to the object centroid.
    """

    def __init__(self, grasp_plan: GraspPlan):
        self.grasp_plan = grasp_plan

    def plan_grasp(self, object_centroid: np.ndarray) -> List[np.ndarray]:
        """
        Generate a sequence of world-frame poses for the grasp.
        
        Args:
            object_centroid: 3D position of the object (3,)
            
        Returns:
            List of poses (7,) [x, y, z, qx, qy, qz, qw]
        """
        world_poses = []
        for wp in self.grasp_plan.waypoints:
            # Transform relative position to world position
            world_pos = object_centroid + wp.position
            
            # Orientation is already in xyzw quaternion
            pose = np.concatenate([world_pos, wp.orientation_quat])
            world_poses.append(pose)
            
        return world_poses
