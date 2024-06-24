# coding=utf-8
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from ip_adapter.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
from PIL import Image, ImageDraw
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
import cv2
from diffusers.utils.import_utils import is_xformers_available
from numpy.linalg import lstsq

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

logger = get_logger(__name__, log_level="INFO")

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="yisol/IDM-VTON", required=False)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--category", type=str, default="upper_body", choices=["upper_body", "lower_body", "dresses"])
    parser.add_argument("--unpaired", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--person_image", type=str, required=True, help="Path to the person image")
    parser.add_argument("--garment_image", type=str, required=True, help="Path to the garment image")
    args = parser.parse_args()

    return args

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images

class SingleTestDataset(data.Dataset):
    def __init__(self, person_image_path: str, garment_image_path: str, category="upper_body", size: Tuple[int, int] = (512, 384)):
        super(SingleTestDataset, self).__init__()
        self.person_image_path = person_image_path
        self.garment_image_path = garment_image_path
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.toTensor = transforms.ToTensor()
        self.category = category

    def __getitem__(self, index):
        person_image = Image.open(self.person_image_path).resize((self.width, self.height))
        garment_image = Image.open(self.garment_image_path)

        person_image = self.transform(person_image)
        garment_image = self.transform(garment_image)

        result = {
            "person_image": person_image,
            "garment_image": garment_image,
        }

        return result

    def __len__(self):
        return 1

def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=accelerator_project_config)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained("yisol/IDM-VTON-DC", subfolder="unet", torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=torch.float16)
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet_encoder", torch_dtype=torch.float16)
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=None, use_fast=False)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    UNet_Encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    UNet_Encoder.to(accelerator.device, weight_dtype)
    unet.eval()
    UNet_Encoder.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    test_dataset = SingleTestDataset(person_image_path=args.person_image, garment_image_path=args.garment_image, category=args.category, size=(args.height, args.width))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=4)

    pipe = TryonPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    ).to(accelerator.device)
    pipe.unet_encoder = UNet_Encoder

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for sample in test_dataloader:
                prompt = "model is wearing a garment"
                num_prompts = 1
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * num_prompts

                with torch.inference_mode():
                    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
                        prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)

                generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=sample['person_image'].to(accelerator.device),
                    text_embeds_cloth=prompt_embeds,
                    cloth=sample["garment_image"].to(accelerator.device),
                    mask_image=sample['person_image'].to(accelerator.device),  # This might need to be adjusted based on your mask requirements
                    image=(sample['person_image']+1.0)/2.0,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                )[0]

                for i in range(len(images)):
                    x_sample = pil_to_tensor(images[i])
                    torchvision.utils.save_image(x_sample, os.path.join(args.output_dir, f"result_{i}.png"))

if __name__ == "__main__":
    main()
