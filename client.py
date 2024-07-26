import requests
import numpy as np
from PIL import Image

# Assuming the server is running on localhost:8000
url = "http://localhost:8000/generate"

# Load and prepare the image
image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
image = np.array(Image.open(requests.get(image_url, stream=True).raw).resize((256, 256)))

# Example action
example_action = [0, 0.00784314, 0, 0, 0, 0.04705882, -0.99607843]

# Send request to the server
response = requests.post(url, json={"image": image.tolist(), "action": example_action})

if response.status_code == 200:
    output_image = np.array(response.json()["output_image"])
    Image.fromarray(output_image).save("output_image.jpg")
else:
    print(f"Error: {response.json()['error']}")