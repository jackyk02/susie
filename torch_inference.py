from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
import requests
import time
import os

if not os.path.exists("pt"):
    pipe = UNet2DConditionModel.from_pretrained(
        "jackyk07/pix2pix",
        subfolder="unet",
        in_channels=8,
        safety_checker=None,
        from_flax=True
    ).to("cpu")

    pipe.save_pretrained("pt")


# Load UNet model with flash attention
unet = UNet2DConditionModel.from_pretrained(
    "pt",
    safety_checker=None,
    torch_dtype=torch.float16,
    use_flash_attention=True
).to("cuda")

# Load VAE, text encoder, and tokenizer from runwayml/stable-diffusion-v1-5
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae",
    torch_dtype=torch.float16
)

text_encoder = CLIPTextModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="text_encoder",
    torch_dtype=torch.float16
)
tokenizer = CLIPTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="tokenizer"
)

# Create a scheduler
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

# Construct InstructPix2Pix pipeline
pipe = StableDiffusionInstructPix2PixPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
).to("cuda")

# Set torch dtype
pipe.to(torch_dtype=torch.float16)

# Enable memory efficient attention
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

# Rest of your code remains the same
prompt = "what would it look like after taking the action sharetheteredairspace cae fresher telethon spool?"
image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).resize((256, 256))

num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

# Run inference
start_time = time.time()
output_image = pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    image_guidance_scale=image_guidance_scale
).images[0]
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time of sample_fn: {execution_time:.4f} seconds")

# Save or display the output image
output_image.save("output_image.png")


# Load UNet model with flash attention
unet = UNet2DConditionModel.from_pretrained(
    "pt",
    safety_checker=None,
    torch_dtype=torch.float16,
    use_flash_attention=True
).to("cuda")

# Load VAE, text encoder, and tokenizer from runwayml/stable-diffusion-v1-5
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae",
    torch_dtype=torch.float16
)

text_encoder = CLIPTextModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="text_encoder",
    torch_dtype=torch.float16
)
tokenizer = CLIPTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="tokenizer"
)

# Create a scheduler
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

# Construct InstructPix2Pix pipeline
pipe = StableDiffusionInstructPix2PixPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
).to("cuda")

# Set torch dtype
pipe.to(torch_dtype=torch.float16)

# Enable memory efficient attention
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

# Rest of your code remains the same
prompt = "what would it look like after taking the action sharetheteredairspace cae fresher telethon spool?"
image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).resize((256, 256))

num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

step = 0
while step < 10:
    # Run inference
    start_time = time.time()
    output_image = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale
    ).images[0]
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time of sample_fn: {execution_time:.4f} seconds")
    step += 1

# Save or display the output image
output_image.save("output_image.png")


# Convert from flax first
# from diffusers import UNet2DConditionModel
# pipe = UNet2DConditionModel.from_pretrained("jackyk07/pix2pix", safety_checker=None, from_flax=True).to("cpu")
# pipe.save_pretrained("pt")
