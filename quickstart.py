from susie.model import create_sample_fn
from susie.jax_utils import initialize_compilation_cache
import requests
import numpy as np
from PIL import Image

initialize_compilation_cache()
sample_fn = create_sample_fn("jackyk07/pix2pix")
image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
image = np.array(Image.open(requests.get(
    image_url, stream=True).raw).resize((256, 256)))

example_action = [0, 0.00784314, 0, 0, 0, 0.04705882, -0.99607843]

image_out = sample_fn(image, example_action)

# to display the images if you're in a Jupyter notebook
Image.fromarray(image).save("in_image.jpg")
Image.fromarray(image_out).save("output_image.jpg")
