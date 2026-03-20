import torch
from typing import Optional
from dataclasses import dataclass
from dream2flow.flow.object_flow import ObjectFlow

@dataclass
class ObjectFlowResult:
    """
    Result of object flow generation.
    """
    flow: ObjectFlow

    # Image coordinates of the points tracked in the initial frame
    initial_pixel_coords: torch.Tensor
    
    # Optional binary mask of the tracked object region in the initial frame
    region_mask: Optional[torch.Tensor] = None
    
    # Optional 2D pixel trajectory for each point across frames
    # Shape: (num_frames, num_points, 2) where last dimension is (x, y) pixel coordinates
    pixel_trajectories: Optional[torch.Tensor] = None

    def save(self, filepath: str) -> None:
        """
        Save object flow generation result to a file.
        """
        save_dict = {
            "flow": self.flow,
            "initial_pixel_coords": self.initial_pixel_coords
        }
        
        if self.region_mask is not None:
            save_dict["region_mask"] = self.region_mask
            
        if self.pixel_trajectories is not None:
            save_dict["pixel_trajectories"] = self.pixel_trajectories
            
        torch.save(save_dict, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "ObjectFlowResult":
        """
        Load object flow generation result from a file.
        """
        load_dict = torch.load(filepath, map_location=device, weights_only=False)
        
        return cls(
            flow=load_dict["flow"],
            initial_pixel_coords=load_dict["initial_pixel_coords"],
            region_mask=load_dict.get("region_mask"),
            pixel_trajectories=load_dict.get("pixel_trajectories")
        )
    
    def clone(self) -> "ObjectFlowResult":
        """
        Clone the object flow result.
        """
        return ObjectFlowResult(
            flow=self.flow.clone(),
            initial_pixel_coords=self.initial_pixel_coords.clone(),
            region_mask=self.region_mask.clone() if self.region_mask is not None else None,
            pixel_trajectories=self.pixel_trajectories.clone() if self.pixel_trajectories is not None else None
        )
