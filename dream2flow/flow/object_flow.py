from tensordict import tensorclass
import torch
from typing import Optional
import torch.nn.functional as F

@tensorclass
class ObjectFlow:
    """
    A class representing 3D object flow (formerly Particles).

    Args:
        position: The position of the flow points, shape (T, N, 3)
        valid_mask: The mask of valid points, shape (T, N)
        rgb: The RGB values of the points, shape (T, N, 3)
        normals: The normals of the points, shape (T, N, 3)
    """
    position: torch.Tensor
    rgb: Optional[torch.Tensor] = None
    normals: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.valid_mask is None:
            self.reset_valid_mask()

        # Set the batch size for tensordict operations
        # T is the batch dimension
        self.batch_size = [self.position.shape[0]]

    def reset_valid_mask(self):
        """
        Resets the valid mask to all True.
        """
        self.valid_mask = torch.ones_like(self.position[..., 0], dtype=torch.bool)

    def initialize_color(self, color_rgb: Optional[torch.Tensor] = None) -> None:
        """
        Initialize colors for flow points.

        Args:
            color_rgb (torch.Tensor, optional): RGB color to use for all points. 
                                               If None, defaults to a rainbow-like distribution.
        """
        T, N, _ = self.position.shape
        
        if color_rgb is not None:
            if color_rgb.dim() == 1:
                colors = color_rgb.view(1, 1, 3).repeat(T, N, 1)
            else:
                colors = color_rgb # Assume already correctly shaped
        else:
            # Simple rainbow-like distribution across particles
            # Using a simple hue-based generation without matplotlib to minimize dependencies
            hues = torch.linspace(0, 1, N)
            # Simple HSV to RGB conversion (simplified)
            def hsv_to_rgb(h):
                r = torch.clamp(torch.abs(h * 6 - 3) - 1, 0, 1)
                g = torch.clamp(2 - torch.abs(h * 6 - 2), 0, 1)
                b = torch.clamp(2 - torch.abs(h * 6 - 4), 0, 1)
                return torch.stack([r, g, b], dim=-1)
            
            particle_colors = hsv_to_rgb(hues) # (N, 3)
            colors = particle_colors.unsqueeze(0).repeat(T, 1, 1)

        self.rgb = colors.to(self.position.device)

    def visualize(self, viewer, name: str, **kwargs):
        """
        Visualizes the object flow in the 3D viewer.
        """
        viewer.visualize_object_flow(self, name, **kwargs)

    def save(self, filepath: str) -> None:
        """
        Save flow data to a file.
        """
        save_dict = {
            "position": self.position,
            "valid_mask": self.valid_mask,
            "rgb": self.rgb,
            "normals": self.normals,
        }
        torch.save(save_dict, filepath)

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> "ObjectFlow":
        """
        Load flow data from a file.
        """
        load_dict = torch.load(filepath, map_location=device)
        return cls(
            position=load_dict["position"],
            valid_mask=load_dict["valid_mask"],
            rgb=load_dict.get("rgb"),
            normals=load_dict.get("normals"),
        )

    def pad(self, target_size: int, dim: int = 1) -> "ObjectFlow":
        """
        Pad the flow tensors with zeros along the specified dimension up to target_size.
        """
        current_size = self.position.shape[dim]
        if current_size >= target_size:
            return self

        pad_size = target_size - current_size
        
        ndim = len(self.position.shape)
        padding = [0, 0] * ndim
        # F.pad uses (last_dim_left, last_dim_right, second_last_dim_left, ...)
        # So for dim=1 in (T, N, 3), it's the second dimension from the left, 
        # which is the second dimension from the right.
        padding[2 * (ndim - 1 - dim) + 1] = pad_size
        padding = tuple(padding)

        padded_position = F.pad(self.position, padding)
        padded_rgb = F.pad(self.rgb, padding) if self.rgb is not None else None
        padded_normals = F.pad(self.normals, padding) if self.normals is not None else None

        # For valid_mask (T, N), dim=1 is the last dimension
        mask_padding = [0, 0] * len(self.valid_mask.shape)
        mask_padding[2 * (len(self.valid_mask.shape) - 1 - dim) + 1] = pad_size
        padded_valid_mask = F.pad(self.valid_mask.float(), tuple(mask_padding), value=0.0).bool()

        return ObjectFlow(
            position=padded_position,
            valid_mask=padded_valid_mask,
            rgb=padded_rgb,
            normals=padded_normals,
        )
