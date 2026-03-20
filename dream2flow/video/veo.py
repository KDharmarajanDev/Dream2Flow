import time
import torch
import numpy as np
from dream2flow.video.base import VideoSource
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from typing import Optional
import cv2

class VeoVideoSource(VideoSource):
    """
    Video source that uses Google's Gemini API (Veo 3) for image-to-video generation.
    """
    def __init__(self, veo_model: str = "veo-3.1-fast-generate-preview", device: str = "cpu") -> None:
        super().__init__(device)
        load_dotenv()
        self.client = None
        self.model = veo_model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY must be set in your environment or .env file for VeoVideoSource.")

    def _load_client(self) -> None:
        self.client = genai.Client(api_key=self.api_key)

    def generate_video(
        self,
        output_dir: str,
        text_prompt: str,
        start_image: torch.Tensor,
        end_image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a video using Gemini API (Veo 3) from a text prompt and a start image.
        """
        if self.client is None:
            self._load_client()

        # Convert start_image to numpy and uint8
        if isinstance(start_image, torch.Tensor):
            img = start_image.detach().cpu().numpy()
        else:
            img = start_image
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Gemini API expects PIL Image
        pil_img = Image.fromarray(img)
        image_bytes_io = BytesIO()
        IMG_FORMAT = "PNG"
        pil_img.save(image_bytes_io, format=IMG_FORMAT)
        image_bytes = image_bytes_io.getvalue()

        # Prepare the prompt and image
        operation = self.client.models.generate_videos(
            model=self.model,
            prompt=text_prompt,
            image=types.Image(image_bytes=image_bytes, mime_type=IMG_FORMAT),
            config=types.GenerateVideosConfig(
                number_of_videos=1,
                negative_prompt="fast motion, camera motion",
            )
        )

        start_time = time.time()
        poll_interval = 10
        last_print = 0
        while not operation.done:
            elapsed = int(time.time() - start_time)
            if elapsed - last_print >= poll_interval:
                print(f"[VeoVideoSource] Waiting for video generation... {elapsed} seconds elapsed.")
                last_print = elapsed
            time.sleep(10)
            operation = self.client.operations.get(operation)

        # Download the generated video
        try:
            generated_video = operation.response.generated_videos[0]
            video_path = os.path.join(output_dir, "rgb.mp4")
            os.makedirs(output_dir, exist_ok=True)
            self.client.files.download(file=generated_video.video)
            generated_video.video.save(video_path)
            print(f"[VeoVideoSource] Generated video saved to {video_path}")
        except Exception as e:
            print(f"[VeoVideoSource] Error downloading video: {e}")
            raise e

        # Read video into torch.Tensor (T, H, W, 3)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        
        if not frames:
            raise RuntimeError(f"No frames found in generated video: {video_path}")
            
        video_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
        return video_tensor.to(self.device)
