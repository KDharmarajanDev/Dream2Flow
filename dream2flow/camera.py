import torch
import json
from typing import Dict, Tuple, Optional

class CameraCalibration:
    """
    Class for managing camera calibration information including intrinsics and extrinsics matrices.
    """

    def __init__(self, camera_name_to_intrinsics_extrinsics: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Initialize camera calibration info.

        Args:
            camera_name_to_intrinsics_extrinsics (Dict[str, Tuple[torch.Tensor, torch.Tensor]], optional): 
                Dictionary mapping camera names to their intrinsics and extrinsics matrices.
        """
        self._camera_name_to_intrinsics_extrinsics = camera_name_to_intrinsics_extrinsics or {}

    def get_camera_calibration(self, camera_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get camera intrinsics and extrinsics for a given camera name.

        Args:
            camera_name (str): Name of the camera to get calibration for

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (camera intrinsics matrix, camera extrinsics matrix)

        Raises:
            ValueError: If camera name is not found
        """
        if camera_name not in self._camera_name_to_intrinsics_extrinsics:
            raise ValueError(f"Camera {camera_name} not found in calibration data")
        
        return self._camera_name_to_intrinsics_extrinsics[camera_name]

    def add_camera_calibration(self, camera_name: str, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        """
        Add or update camera calibration data for a specific camera.

        Args:
            camera_name (str): Name of the camera
            intrinsics (torch.Tensor): Camera intrinsics matrix
            extrinsics (torch.Tensor): Camera extrinsics matrix
        """
        self._camera_name_to_intrinsics_extrinsics[camera_name] = (intrinsics, extrinsics)

    def save(self, filepath: str):
        """
        Save camera calibration data to a file.

        Args:
            filepath (str): Path to save the calibration data
        """
        # Convert tensors to lists for JSON serialization
        calibration_data = {}
        for camera_name, (intrinsics, extrinsics) in self._camera_name_to_intrinsics_extrinsics.items():
            calibration_data[camera_name] = {
                'intrinsics': intrinsics.cpu().tolist(),
                'extrinsics': extrinsics.cpu().tolist()
            }

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=4)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> 'CameraCalibration':
        """
        Load camera calibration data from a file.

        Args:
            filepath (str): Path to load the calibration data from
            device (str): Device to load the calibration data to

        Returns:
            CameraCalibration: New instance with loaded calibration data
        """
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)

        # Convert lists back to tensors
        camera_name_to_intrinsics_extrinsics = {}
        for camera_name, data in calibration_data.items():
            intrinsics = torch.tensor(data['intrinsics']).to(device)
            extrinsics = torch.tensor(data['extrinsics']).to(device)
            camera_name_to_intrinsics_extrinsics[camera_name] = (intrinsics, extrinsics)

        return cls(camera_name_to_intrinsics_extrinsics)

    @classmethod
    def from_dict(cls, data: Dict, device: str = "cpu") -> 'CameraCalibration':
        """
        Create CameraCalibration from a dictionary.
        """
        camera_name_to_intrinsics_extrinsics = {}
        for camera_name, calib in data.items():
            intrinsics = torch.tensor(calib['intrinsics']).to(device)
            extrinsics = torch.tensor(calib['extrinsics']).to(device)
            camera_name_to_intrinsics_extrinsics[camera_name] = (intrinsics, extrinsics)
        return cls(camera_name_to_intrinsics_extrinsics)

    def visualize(self, viewer):
        """
        Visualizes the camera extrinsics and intrinsics by showing a camera frustum for each camera.
        """
        for camera_name, (intrinsics, extrinsics) in self._camera_name_to_intrinsics_extrinsics.items():
            viewer.register_camera(camera_name, intrinsics, extrinsics)
