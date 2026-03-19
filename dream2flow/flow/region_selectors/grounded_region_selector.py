from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

from dream2flow.flow.region_selectors.instruction_processor import InstructionProcessor


@dataclass
class GroundedRegionSelectorConfig:
    save_intermediate_results: bool = True
    enable_instruction_processing: bool = True
    grounding_dino_box_threshold: float = 0.4
    grounding_dino_text_threshold: float = 0.3
    grounding_dino_model_path: str = "IDEA-Research/grounding-dino-base"
    sam2_model_path: str = "facebook/sam2-hiera-large"


class GroundedRegionSelector:
    def __init__(self, config: GroundedRegionSelectorConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device

        if self.config.enable_instruction_processing:
            self._instruction_processor = InstructionProcessor()
        else:
            self._instruction_processor = None

        workspace_root = Path(__file__).resolve().parents[4]
        sam2_root = workspace_root / "video-particles" / "deps" / "sam2"
        if str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))

        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from torchvision import transforms
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._grounding_dino_processor = AutoProcessor.from_pretrained(
            self.config.grounding_dino_model_path,
            local_files_only=True,
        )
        self._grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.grounding_dino_model_path,
            local_files_only=True,
        ).to(self.device)

        GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")
        self._sam2_image_predictor = SAM2ImagePredictor.from_pretrained(
            self.config.sam2_model_path,
            device=self.device,
        )
        self._torch_to_pil_image = transforms.ToPILImage()

    def _process_instruction_for_grounding_dino(self, instruction: str) -> str:
        if self._instruction_processor is None:
            return instruction
        return self._instruction_processor.process_for_grounding_dino(instruction)

    def _get_object_bbox(self, image: Image.Image, instruction: str) -> torch.Tensor:
        processed_instruction = self._process_instruction_for_grounding_dino(instruction)
        grounding_dino_inputs = self._grounding_dino_processor(
            text=processed_instruction,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            grounding_dino_outputs = self._grounding_dino_model(**grounding_dino_inputs)

        bbox_results = self._grounding_dino_processor.post_process_grounded_object_detection(
            grounding_dino_outputs,
            grounding_dino_inputs.input_ids,
            box_threshold=self.config.grounding_dino_box_threshold,
            text_threshold=self.config.grounding_dino_text_threshold,
            target_sizes=[image.size[::-1]],
        )
        return bbox_results[0]["boxes"][0]

    def _visualize_bounding_box(self, image: Image.Image, bbox: torch.Tensor) -> Image.Image:
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        draw.rectangle(bbox.cpu().tolist(), outline="red", width=3)
        return draw_image

    def extract_region(self, output_dir: str, image: torch.Tensor, instruction: str) -> torch.Tensor:
        if image.dim() == 3 and image.shape[-1] in (1, 3, 4):
            image_pil = self._torch_to_pil_image(image.permute(2, 0, 1).cpu())
        else:
            image_pil = self._torch_to_pil_image(image.cpu())

        bbox = self._get_object_bbox(image_pil, instruction)
        output_dir_path = Path(output_dir)

        if self.config.save_intermediate_results:
            draw_image = self._visualize_bounding_box(image_pil, bbox)
            draw_image.save(output_dir_path / "region_bbox.png")

        if self.device.startswith("cuda"):
            autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext

            autocast_context = nullcontext()

        with torch.inference_mode(), autocast_context:
            self._sam2_image_predictor.set_image(image_pil)
            masks, _, _ = self._sam2_image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox,
                multimask_output=False,
            )

        mask = torch.from_numpy(masks).to(device=self.device)
        if self.config.save_intermediate_results:
            mask_uint8 = (mask.squeeze(0).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(mask_uint8).save(output_dir_path / "region_mask.png")

        return mask.squeeze(0).bool()
