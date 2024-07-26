from susie.jax_utils import initialize_compilation_cache
from susie.model import create_sample_fn
import os
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

import json_numpy
json_numpy.patch()


# Initialize the compilation cache
initialize_compilation_cache()


class Pix2PixServer:
    def __init__(self, model_path: str):
        """
        A simple server for Pix2Pix models; exposes `/generate` to create an image given an input image and action.
        => Takes in {"image": np.ndarray, "action": np.ndarray}
        => Returns  {"output_image": np.ndarray}
        """
        self.model_path = model_path
        self.sample_fn = create_sample_fn(self.model_path)

    def generate_image(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, action = payload["image"], payload["action"]
            output_image = self.sample_fn(image, action)

            if double_encode:
                return JSONResponse(json_numpy.dumps(output_image))
            else:
                return JSONResponse(output_image)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'action': np.ndarray}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/generate")(self.generate_image)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    model_path: str = "jackyk07/pix2pix"  # Path to the Pix2Pix model
    host: str = "0.0.0.0"                 # Host IP Address
    port: int = 8000                      # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = Pix2PixServer(cfg.model_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
