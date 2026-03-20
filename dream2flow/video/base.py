import torch
from abc import ABC, abstractmethod
from typing import Optional

class VideoSource(ABC):
    """
    Base class for video sources.
    """
    
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
    
    @abstractmethod
    def generate_video(
        self, 
        output_dir: str,
        text_prompt: str, 
        start_image: torch.Tensor,
        end_image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a video based on a text prompt and start image.
        
        Args:
            output_dir: Directory to save results
            text_prompt (str): Text description of the desired video
            start_image (torch.Tensor): RGB start image as torch Tensor (H, W, 3)
            end_image (Optional[torch.Tensor]): Optional RGB end image as torch Tensor (H, W, 3)
            **kwargs: Additional arguments for video generation
            
        Returns:
            torch.Tensor: Video frames as torch Tensor (T, H, W, 3) where T is the number of frames
        """
        pass
