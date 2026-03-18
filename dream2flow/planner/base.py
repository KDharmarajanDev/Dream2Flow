from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from dream2flow.flow.object_flow_result import ObjectFlowResult

@dataclass
class PlanResult:
    """
    Result of motion planning.
    """
    # Optimized joint configurations, shape (T, num_joints)
    joint_trajectory: np.ndarray
    
    # End-effector poses in xyzw quaternion format, shape (T, 7)
    # Derived from joint_trajectory via forward kinematics
    ee_trajectory: np.ndarray

class Planner(ABC):
    """
    Base class for motion planners.
    """
    
    @abstractmethod
    def plan(self, 
             flow_result: ObjectFlowResult, 
             initial_joints: np.ndarray, 
             initial_pose: np.ndarray,
             **kwargs) -> PlanResult:
        """
        Plan a trajectory based on object flow.
        
        Args:
            flow_result: Result of object flow generation
            initial_joints: Initial joint positions (num_joints,)
            initial_pose: Initial end-effector pose (7,) (x, y, z, qx, qy, qz, qw)
            **kwargs: Additional arguments
            
        Returns:
            PlanResult: Result of motion planning
        """
        pass
