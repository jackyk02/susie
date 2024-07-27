from diffusers import UNet2DConditionModel
pipe = UNet2DConditionModel.from_pretrained("jackyk07/pix2pix", safety_checker=None, from_flax=True).to("cpu")
pipe.save_pretrained("pt")

