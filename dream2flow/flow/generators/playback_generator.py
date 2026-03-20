import os
import torch
from dream2flow.flow.generators.base import ObjectFlowGenerator
from dream2flow.flow.object_flow_result import ObjectFlowResult
from dream2flow.camera import CameraCalibration

class PlaybackFlowGenerator(ObjectFlowGenerator):
    """
    Generates object flow by loading it from a pre-saved file.
    """

    def __init__(self, trajectory_filepath: str):
        self.trajectory_filepath = trajectory_filepath

    def generate(self, 
                 output_dir: str, 
                 start_image: torch.Tensor, 
                 camera: CameraCalibration,
                 camera_name: str,
                 **kwargs) -> ObjectFlowResult:
        """
        Loads an object flow result from a file.
        """
        if not os.path.exists(self.trajectory_filepath):
            raise FileNotFoundError(f"Flow result file not found: {self.trajectory_filepath}")
        
        return ObjectFlowResult.load(self.trajectory_filepath)
