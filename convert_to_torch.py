from diffusers import UNet2DConditionModel

pipe = UNet2DConditionModel.from_pretrained(
    'jackyk07/pix2pix', subfolder='unet', safety_checker=None, from_flax=True
)

pipe.save_pretrained("pt")


# from diffusers import UNet2DConditionModel
# pipe = UNet2DConditionModel.from_pretrained("pt", safety_checker=None).to("cuda")
# image = pipe("duck", num_inference_steps=50).images[0]
# display(image)
