import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
from typing import Any
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
import safetensors.torch
import bitsandbytes as bnb

from preprocess import KyrgyzOrnamentDataset


def train_lora(
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        annotations_path: str = "../data/annotations.csv",
        images_root: str = "../data",
        output_path: str = "kyrgyz_lora.safetensors",
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        num_steps: int = 300,
        learning_rate: float = 1e-4,
        rank: int = 16,
        alpha: int = 16
):
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16"
    )
    device = accelerator.device

    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    tokenizer: Any = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    tokenizer_2: Any = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_2.pad_token = tokenizer_2.eos_token

    unet: Any = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    unet.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config, autocast_adapter_dtype=False)

    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    unet.print_trainable_parameters()

    dataset = KyrgyzOrnamentDataset(annotations_path, images_root, tokenizer=None)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(num_steps * 0.1),
        num_training_steps=num_steps
    )

    # noinspection PyTypeChecker
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)

    noise_scheduler: Any = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    vae: Any = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device, dtype=weight_dtype)
    text_encoder: Any = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device,
                                                                                               dtype=weight_dtype)
    text_encoder_2: Any = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2").to(device,
                                                                                                                 dtype=weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    sample_img = dataset[0]["pixel_values"]
    img_h, img_w = sample_img.shape[1], sample_img.shape[2]
    time_ids = torch.tensor([[img_h, img_w, 0, 0, img_h, img_w]], device=device, dtype=weight_dtype).repeat(batch_size,
                                                                                                            1)

    global_step = 0
    progress_bar = tqdm(range(num_steps), desc="Train LoRA", disable=not accelerator.is_local_main_process)
    unet.train()

    num_train_timesteps = getattr(noise_scheduler.config, "num_train_timesteps", 1000)

    while global_step < num_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                pixel_values = 2.0 * pixel_values - 1.0

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompts = batch["prompt"]
                with torch.no_grad():
                    tokens_1 = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt").input_ids.to(device)
                    encoder_hidden_states_1 = text_encoder(tokens_1).last_hidden_state

                    tokens_2 = tokenizer_2(prompts, padding="max_length", max_length=tokenizer_2.model_max_length,
                                           truncation=True, return_tensors="pt").input_ids.to(device)
                    encoder_output_2 = text_encoder_2(tokens_2)
                    encoder_hidden_states_2 = encoder_output_2.last_hidden_state
                    pooled_output_2 = encoder_output_2.text_embeds

                encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                added_cond_kwargs = {"text_embeds": pooled_output_2, "time_ids": time_ids[:latents.shape[0]]}

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                                  added_cond_kwargs=added_cond_kwargs).sample

                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.6f}")

                if global_step >= num_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state_dict = {k: v for k, v in unwrapped_unet.state_dict().items() if "lora" in k}
        safetensors.torch.save_file(lora_state_dict, output_path)
        print(f"\nSuccess! LoRA weights saved to: {output_path}")


if __name__ == "__main__":
    train_lora()