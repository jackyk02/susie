from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
import requests
import time
import os


def load_or_create_unet():
    if not os.path.exists("pt"):
        unet = UNet2DConditionModel.from_pretrained(
            "jackyk07/pix2pix", subfolder="unet", in_channels=8,
            safety_checker=None, from_flax=True
        ).to("cpu")
        unet.save_pretrained("pt")
    return UNet2DConditionModel.from_pretrained(
        "pt", safety_checker=None, torch_dtype=torch.float16,
        use_flash_attention=True
    ).to("cuda")


def load_models():
    unet = load_or_create_unet()
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    return unet, vae, text_encoder, tokenizer, scheduler


def create_pipeline(unet, vae, text_encoder, tokenizer, scheduler):
    pipe = StableDiffusionInstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False
    ).to("cuda")
    pipe.to(torch_dtype=torch.float16)
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def run_inference(pipe, prompt, image, num_inference_steps, guidance_scale, image_guidance_scale):
    start_time = time.time()
    output_image = pipe(
        prompt=prompt, image=image, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale
    ).images[0]
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    return output_image


def main():
    unet, vae, text_encoder, tokenizer, scheduler = load_models()
    pipe = create_pipeline(unet, vae, text_encoder, tokenizer, scheduler)

    prompt = "what would it look like after taking the action sharetheteredairspace cae fresher telethon spool?"
    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
    image = Image.open(requests.get(
        image_url, stream=True).raw).resize((256, 256))

    for _ in range(10):
        output_image = run_inference(pipe, prompt, image, 20, 10, 1.5)

    output_image.save("output_image.png")


if __name__ == "__main__":
    main()
