from PIL import Image
import requests
import json_numpy
import numpy as np
json_numpy.patch()

# Load and prepare the image
image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
image = np.array(Image.open(requests.get(
    image_url, stream=True).raw).resize((256, 256)), dtype=np.uint8)

# Example action
example_action = np.array([0, 0.00784314, 0, 0, 0, 0.04705882, -0.99607843])

next_obs = requests.post(
    "http://0.0.0.0:8000/generate",
    json={"image": image, "action": example_action}
).json()

print(next_obs.shape)

Image.fromarray(next_obs).save("next_obs.jpg")
