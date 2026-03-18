from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict
from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.camera import CameraCalibration

class ObjectFlowGenerator(ABC):
    """Abstract base class for generating 3D object flow."""

    @abstractmethod
    def generate(self, 
                 output_dir: str, 
                 start_image: torch.Tensor, 
                 camera: CameraCalibration,
                 camera_name: str,
                 **kwargs) -> ObjectFlowResult:
        """
        Generates 3D object flow.

        Args:
            output_dir: Directory to save results
            start_image: Initial RGB image (H, W, 3)
            camera: Camera calibration information
            camera_name: Name of the camera being used
            **kwargs: Additional arguments
            
        Returns:
            ObjectFlowResult: Result of object flow generation
        """
        pass
