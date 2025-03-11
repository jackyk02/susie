import os
from functools import partial
from typing import Any, Dict, List
from ml_collections import ConfigDict
from copy import deepcopy

import dlimp as dl
import numpy as np
import tensorflow as tf
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
import jax
import goal_relabeling
from transformers import CLIPTokenizer
from image_transform import process_image
from PIL import Image
import random
import pickle

seed = 88
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
jax.random.PRNGKey(seed)

import re
import string
import nltk
from nltk.corpus import words

# Download NLTK words corpus if not already downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Load English words into a set for faster lookups
english_words = set(word.lower() for word in words.words())

def preprocess_instruction(instruction):
    """
    Preprocess the instruction string:
    1. Check if it contains numerical digits
    2. Check if it contains special characters (anything not a letter, space, or punctuation)
    3. Convert to lowercase
    4. Remove punctuation
    5. Normalize spaces between words to exactly one space
    6. Verify it contains at least one valid English word
    
    Returns preprocessed string if valid, None if should be skipped
    """
    # Skip empty instructions
    if not instruction or instruction.strip() == "":
        return None
    
    # Check for numerical digits
    if any(char.isdigit() for char in instruction):
        return None
    
    # Define allowed characters (letters, spaces, and standard punctuation)
    allowed_chars = string.ascii_letters + string.punctuation + ' '
    
    # Check for special characters
    if any(char not in allowed_chars for char in instruction):
        return None
    
    # Convert to lowercase
    instruction = instruction.lower()
    
    # Remove punctuation
    instruction = re.sub(r'[^\w\s]', '', instruction)
    
    # Normalize whitespace to ensure exactly one space between words
    instruction = ' '.join(instruction.split())
    
    # Check if the instruction contains at least one valid English word
    instruction_words = instruction.split()
    if not any(word in english_words for word in instruction_words):
        return None
    
    return instruction

class Transforms:
    """Trajectory-level transforms for each dataset"""
    @staticmethod
    def bridge(x: Dict[str, Any]) -> Dict[str, Any]:
        CAMERA_VIEWS = {"images0", "images1", "images2"}
        # pick a random camera view
        views = tf.stack([x["obs"][k] for k in CAMERA_VIEWS])
        lengths = tf.stack([tf.strings.length(x["obs"][k][0])
                           for k in CAMERA_VIEWS])
        views = views[lengths > 0]
        idx = tf.random.uniform(
            [], minval=0, maxval=tf.shape(views)[0], dtype=tf.int32)
        x["obs"] = views[idx]
        return x


class GetPaths:
    """Retrieves paths to TFRecord files or each dataset"""

    @staticmethod
    def bridge(data_path: str, train: bool) -> str:
        return f"{data_path}/{'train' if train else 'val'}"


def make_dataset(
    name: str,
    data_path: str,
    image_size: int,
    shuffle_buffer_size: int,
    train: bool,
    goal_relabeling_fn: str,
    goal_relabeling_kwargs: dict = {},
    augment_kwargs: dict = {},
) -> dl.DLataset:
    paths = getattr(GetPaths, name)(data_path, train)
    dataset = (
        dl.DLataset.from_tfrecords(paths)
        .map(dl.transforms.unflatten_dict)
        .map(getattr(Transforms, name))
        .filter(lambda x: tf.math.reduce_all(x["lang"] != ""))
        .apply(
            partial(
                getattr(goal_relabeling, goal_relabeling_fn), **goal_relabeling_kwargs
            ),
        )
        .unbatch()
        .shuffle(shuffle_buffer_size)
    )

    dataset = dataset.map(
        partial(dl.transforms.decode_images, match=["curr"])
    ).map(
        partial(
            dl.transforms.resize_images,
            match=["curr"],
            size=(image_size, image_size),
        )
    )

    if train:
        dataset = dataset.map(
            partial(
                dl.transforms.augment,
                traj_identical=False,
                keys_identical=True,
                match=["curr"],
                augment_kwargs=augment_kwargs,
            )
        )

    # normalize images to [-1, 1]
    dataset = dataset.map(
        partial(
            dl.transforms.selective_tree_map,
            match=["curr"],
            map_fn=lambda v: tf.cast(v, tf.float32) / 127.5 - 1.0,
        )
    )

    return dataset.repeat()


def get_data_loader(data_config, tokenize_fn, mesh=None):
    data_config = dict(data_config)
    batch_size = data_config.pop("batch_size")

    train_datasets = []
    val_datasets = []
    weights = []
    for data_name, data_kwargs in data_config.items():
        data_kwargs = dict(data_kwargs)
        weights.append(float(data_kwargs.pop("weight")))
        train_datasets.append(make_dataset(
            data_name, train=True, **data_kwargs))
        val_datasets.append(make_dataset(
            data_name, train=False, **data_kwargs))

    train = dl.DLataset.sample_from_datasets(
        train_datasets, weights=weights, stop_on_empty_dataset=True
    ).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    val = dl.DLataset.sample_from_datasets(
        val_datasets, weights=weights, stop_on_empty_dataset=True
    ).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch,
            mesh,
            P(("dp", "fsdp")),
        )

    train = map(tokenize_fn, train.as_numpy_iterator())
    val = map(tokenize_fn, val.as_numpy_iterator())

    if mesh:
        return map(shard, train), map(shard, val), len(train_datasets)
    else:
        return train, val, len(train_datasets)


def tokenize_fn(batch):
    tokenizer = CLIPTokenizer.from_pretrained(
        "lodestones/stable-diffusion-v1-5-flax", subfolder="tokenizer"
    )

    def tokenize(s: List[str]) -> np.ndarray:
        return tokenizer(s, padding="max_length", return_tensors="np").input_ids

    lang = [s.decode("utf-8") for s in batch["lang"]]
    assert all(s != "" for s in lang)
    batch["prompt_ids"] = tokenize(lang)
    return batch


def serialize_data(train_samples=70000):
    """
    Process and save data without chunking for instructions and actions.
    Save images as JPG files instead of storing in a dictionary.
    Only processes training data.
    """
    # Create output directories
    os.makedirs("images", exist_ok=True)
    
    config = ConfigDict()
    config.data = ConfigDict()
    config.seed = seed
    config.data.batch_size = 50

    data_base = ConfigDict()
    data_base.image_size = 256
    data_base.shuffle_buffer_size = 100000
    data_base.augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.85, 1.0], ratio=[0.95, 1.05]),
        random_brightness=[0.05],
        random_contrast=[0.95, 1.05],
        random_saturation=[0.95, 1.05],
        random_hue=[0.025],
        augment_order=[],
    )

    config.data.bridge = bridge = deepcopy(data_base)
    bridge.weight = 45.0
    bridge.data_path = "/root/tf_lang"
    bridge.goal_relabeling_fn = "subgoal_only"
    bridge.goal_relabeling_kwargs = dict(
        subgoal_delta=(11, 14),
        truncate=False,
    )

    train_loader, _, _ = get_data_loader(config.data, tokenize_fn)

    # Initialize dictionaries for instructions and actions (not chunked)
    instruction_dict = {}
    action_dict = {}

    # Process training data
    print("Processing training data...")
    processed = 0
    
    while processed < train_samples:
        try:
            data = next(train_loader)
            batch_size = data["curr"].shape[0]

            for index in range(batch_size):
                if processed >= train_samples:
                    break
                instruction = data["lang"][index].decode("utf-8")
                instruction = preprocess_instruction(instruction)
                if instruction is None:
                    continue
                # print(instruction)

                act = data["actions"][index]
                if np.any(act == 0):
                    continue

                input_image = data["curr"][index]
                # Scale image from [-1, 1] to [0, 255]
                scaled_image = np.clip((input_image + 1) * 127.5, 0, 255).astype(np.uint8)
                
                # Save image as JPG using PIL
                img = Image.fromarray(scaled_image)
                img_path = f"images/{processed}.jpg"
                img.save(img_path)

                process_image(
                    img_path,
                    output_dir="./output/",
                    crop_scale=0.9,
                    target_size=(224, 224),
                    batch_size=1
                )
                
                # Store instruction and action in dictionaries
                instruction_dict[processed] = instruction
                action_dict[processed] = act

                processed += 1

                if processed % 1000 == 0:
                    print(f"Processed {processed} training datapoints")

        except Exception as e:
            print(f"Error processing training data: {e}")
            break

    # Save instruction and action dictionaries (not chunked)
    with open("instruction_dict.pkl", "wb") as f:
        pickle.dump(instruction_dict, f)

    with open("action_dict.pkl", "wb") as f:
        pickle.dump(action_dict, f)

    print("\nData serialization complete!")
    print(f"Total samples processed: {processed}")
    print(f"Images saved as JPG files in 'images/' directory")
    print(f"Instruction and action data saved as non-chunked dictionaries")


if __name__ == "__main__":
    # Call the main function with desired sample count
    serialize_data(100)