import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, AutoencoderKL
from peft import LoraConfig, get_peft_model
import safetensors.torch


def apply_symmetry(image: Image.Image, symmetry_type: str) -> Image.Image:
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    if symmetry_type == "bilateral":
        img_np[:, w // 2:] = np.fliplr(img_np[:, :w // 2])
    elif symmetry_type == "radial":
        flipped_h = np.flipud(img_np)
        flipped_w = np.fliplr(img_np)
        flipped_both = np.flipud(np.fliplr(img_np))
        img_np = (img_np.astype(np.uint16) + flipped_h + flipped_w + flipped_both) // 4
        img_np = img_np.astype(np.uint8)
    elif symmetry_type == "ribbon":
        img_np[h // 2:, :] = np.flipud(img_np[:h // 2, :])
    elif symmetry_type == "grid":
        quarter = img_np[:h // 2, :w // 2]
        img_np[:h // 2, w // 2:] = np.fliplr(quarter)
        img_np[h // 2:, :w // 2] = np.flipud(quarter)
        img_np[h // 2:, w // 2:] = np.flipud(np.fliplr(quarter))
    else:
        raise ValueError(f"Unknown symmetry type: {symmetry_type}")

    return Image.fromarray(img_np)


def prepare_control_images(image_path: str):
    """Prepare edge map (Canny) for ControlNet."""
    original_image = cv2.imread(image_path)
    canny_image = cv2.Canny(original_image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    return Image.fromarray(canny_image)


def generate_pattern(
        prompt_a: str,
        prompt_b: str,
        lora_path: str = "1-OTM.safetensors",
        control_image_path: str = None,
        symmetry_type: str = "radial",
        output_path: str = "output.png",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        seed: int = 42
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading base models...")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype
    )

    controlnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=dtype
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet_canny,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if os.path.exists(lora_path):
        print(f"Loading trained LoRA from: {lora_path}...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )

        pipe.unet = get_peft_model(pipe.unet, lora_config, autocast_adapter_dtype=False)

        # Load weights
        state_dict = safetensors.torch.load_file(lora_path)
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        pipe.unet.load_state_dict(state_dict, strict=False)
        # Merge for faster generation
        pipe.unet = pipe.unet.merge_and_unload()
        print("LoRA successfully merged into the model.")
    else:
        print("Warning: LoRA file not found.")

    if device == "cuda":
        pipe.enable_model_cpu_offload()

    full_prompt = f"{prompt_a} | {prompt_b}"

    if control_image_path and os.path.exists(control_image_path):
        print(f"Using control image: {control_image_path}")
        control_image = prepare_control_images(control_image_path)
    else:
        control_image = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))
        controlnet_conditioning_scale = 0.0

    print("Starting generation...")
    generator = torch.Generator(device=device).manual_seed(seed)

    output = pipe(
        prompt=full_prompt,
        image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator
    ).images[0]

    print(f"Applying geometric symmetry: {symmetry_type}")
    symmetric_image = apply_symmetry(output, symmetry_type)

    symmetric_image.save(output_path)
    print(f"Done! Image saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_pattern(
        prompt_a="kyrgyz koshkor-muyuz ornament, traditional shyrdak pattern, flowing curves",
        prompt_b="clean vector lines, high contrast, ornamental balance, no background, symmetrical",
        lora_path="1-OTM.safetensors",
        symmetry_type="grid"
    )