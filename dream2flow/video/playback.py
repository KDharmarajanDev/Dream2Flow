import numpy as np
import cv2
import os
import torch
from typing import Optional
from dream2flow.video.base import VideoSource

class PlaybackVideoSource(VideoSource):
    """
    Video source that plays back a pre-recorded video from a file.
    """
    
    def __init__(self, video_path: str, device: str = "cpu") -> None:
        super().__init__(device)
        self.video_path = video_path
    
    def generate_video(
        self, 
        output_dir: str,
        text_prompt: str, 
        start_image: torch.Tensor,
        end_image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate video by returning the pre-recorded video frames.
        """
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Read video using OpenCV
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {self.video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if not frames:
            raise RuntimeError(f"No frames found in video: {self.video_path}")
        
        video_frames = np.array(frames)
        video_tensor = torch.from_numpy(video_frames).float() / 255.0
        
        return video_tensor.to(self.device)
