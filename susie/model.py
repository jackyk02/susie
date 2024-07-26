import os
import time
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import einops as eo
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import orbax.checkpoint
from absl import logging
from diffusers.models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax.lax import with_sharding_constraint as wsc
from transformers import CLIPTokenizer, FlaxCLIPTextModel

import wandb
from susie import sampling, scheduling, action_processing
from susie.jax_utils import replicate


class EmaTrainState(TrainState):
    params_ema: FrozenDict[str, Any]

    @partial(jax.jit, donate_argnums=0)
    def apply_ema_decay(self, ema_decay):
        params_ema = jax.tree_map(
            lambda p_ema, p: p_ema * ema_decay + p * (1.0 - ema_decay),
            self.params_ema,
            self.params,
        )
        return self.replace(params_ema=params_ema)


def create_model_def(config: dict) -> FlaxUNet2DConditionModel:
    model, unused_kwargs = FlaxUNet2DConditionModel.from_config(
        dict(config), return_unused_kwargs=True
    )
    if unused_kwargs:
        logging.warning(
            f"FlaxUNet2DConditionModel unused kwargs: {unused_kwargs}")
    # monkey-patch __call__ to use channels-last
    model.__call__ = lambda self, sample, *args, **kwargs: eo.rearrange(
        FlaxUNet2DConditionModel.__call__(
            self, eo.rearrange(sample, "b h w c -> b c h w"), *args, **kwargs
        ).sample,
        "b c h w -> b h w c",
    )
    return model


def load_vae(
    path: str,
) -> Tuple[
    Callable[[jax.Array, jax.Array, bool], jax.Array],
    Callable[[jax.Array, bool], jax.Array],
]:
    if ":" in path:
        path, revision = path.split(":")
    else:
        revision = None
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        path, subfolder="vae", revision=revision
    )
    # monkey-patch encode to use channels-last (it returns a FlaxDiagonalGaussianDistribution object, which is already
    # channels-last)
    vae.encode = lambda self, sample, *args, **kwargs: FlaxAutoencoderKL.encode(
        self, eo.rearrange(sample, "b h w c -> b c h w"), *args, **kwargs
    ).latent_dist

    # monkey-patch decode to use channels-last (it already accepts channels-last input)
    vae.decode = lambda self, latents, *args, **kwargs: eo.rearrange(
        FlaxAutoencoderKL.decode(self, latents, *args, **kwargs).sample,
        "b c h w -> b h w c",
    )

    # HuggingFace places vae_params committed onto the CPU -_-
    # this one took me awhile to figure out...
    vae_params = jax.device_get(vae_params)

    @jax.jit
    def vae_encode(vae_params, key, sample, scale=False):
        # handle the case where `sample` is multiple images stacked
        batch_size = sample.shape[0]
        sample = eo.rearrange(sample, "n h w (x c) -> (n x) h w c", c=3)
        latents = vae.apply({"params": vae_params}, sample, method=vae.encode).sample(
            key
        )
        latents = eo.rearrange(
            latents, "(n x) h w c -> n h w (x c)", n=batch_size)
        latents = jax.lax.cond(
            scale, lambda: latents * vae.config.scaling_factor, lambda: latents
        )
        return latents

    @jax.jit
    def vae_decode(vae_params, latents, scale=True):
        # handle the case where `latents` is multiple images stacked
        batch_size = latents.shape[0]
        latents = eo.rearrange(
            latents, "n h w (x c) -> (n x) h w c", c=vae.config.latent_channels
        )
        latents = jax.lax.cond(
            scale, lambda: latents / vae.config.scaling_factor, lambda: latents
        )
        sample = vae.apply({"params": vae_params}, latents, method=vae.decode)
        sample = eo.rearrange(
            sample, "(n x) h w c -> n h w (x c)", n=batch_size)
        return sample

    return partial(vae_encode, vae_params), partial(vae_decode, vae_params)


def load_text_encoder(
    path: str,
) -> Tuple[
    Callable[[List[str]], np.ndarray],
    Callable[[np.ndarray], List[str]],
    Callable[[jax.Array], jax.Array],
]:
    if ":" in path:
        path, revision = path.split(":")
    else:
        revision = None
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        path, subfolder="text_encoder", revision=revision
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        path, subfolder="tokenizer", revision=revision
    )
    # print(tokenizer.max_length)

    action_tokenizer = action_processing.ActionTokenizer(tokenizer)

    # print("begin index: " + str(action_tokenizer.action_token_begin_idx))
    # print("vocab size: " + str(action_tokenizer.tokenizer.vocab_size))
    # print("bin center: " + str(action_tokenizer.bin_centers))
    # example_action = np.array([-1.70204811e-02, -3.27922452e-02, 3.99110917e-02, -4.45905340e-02,
    #                           -8.79744622e-02, -1.08816563e-01, 0.0000012312])
    # the corresponding least used words after calling:ricciardo fresher disintegrrevs linkin stewards airspace
    # print("after calling:" + action_tokenizer.__call__(example_action))

    def tokenize(s: List[str]) -> np.ndarray:
        # print("tokenize input!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: " + str(s))
        # print(tokenizer(s, padding="max_length", return_tensors="np").input_ids)
        # print(tokenizer(s, padding="max_length",
        #      return_tensors="np").input_ids.shape)
        return tokenizer(s, padding="max_length", return_tensors="np").input_ids

    untokenize = partial(tokenizer.batch_decode, skip_special_tokens=True)

    @ jax.jit
    def text_encode(params, prompt_ids):
        return text_encoder(prompt_ids, params=params)[0]

    return tokenize, untokenize, partial(text_encode, text_encoder.params), action_tokenizer

    # def tokenize(s: List[str]) -> np.ndarray:
    #     return action_tokenizer(s)

    # def untokenize(token_ids: np.ndarray) -> List[str]:
    #     return action_tokenizer.decode_token_ids_to_actions(token_ids)

    # @jax.jit
    # def text_encode(params, prompt_ids):
    #     return text_encoder(prompt_ids, params=params)[0]

    # return tokenize, untokenize, partial(text_encode, text_encoder.params)


def load_pretrained_unet(
    path: str, in_channels: int
) -> Tuple[FlaxUNet2DConditionModel, dict]:
    model_def, params = FlaxUNet2DConditionModel.from_pretrained(
        path, dtype=np.float32, subfolder="unet"
    )

    # same issue, they commit the params to the CPU, which totally messes stuff
    # up downstream...
    params = jax.device_get(params)

    # add extra parameters to conv_in if necessary
    old_conv_in = params["conv_in"]["kernel"]
    h, w, cin, cout = old_conv_in.shape
    logging.info(f"Adding {in_channels - cin} channels to conv_in")
    params["conv_in"]["kernel"] = np.zeros(
        (h, w, in_channels, cout), dtype=old_conv_in.dtype
    )
    params["conv_in"]["kernel"][:, :, :cin, :] = old_conv_in

    # monkey-patch __call__ to use channels-last
    model_def.__call__ = lambda self, sample, *args, **kwargs: eo.rearrange(
        FlaxUNet2DConditionModel.__call__(
            self, eo.rearrange(sample, "b h w c -> b c h w"), *args, **kwargs
        ).sample,
        "b c h w -> b h w c",
    )

    return model_def, params


def create_sample_fn(
    path: str,
    wandb_run_name: Optional[str] = None,
    num_timesteps: int = 50,
    prompt_w: float = 7.5,
    context_w: float = 2.5,
    eta: float = 0.0,
    pretrained_path: str = "runwayml/stable-diffusion-v1-5:flax",
) -> Callable[[np.ndarray, str], np.ndarray]:
    if (
        os.path.exists(path)
        and os.path.isdir(path)
        and "checkpoint" in os.listdir(path)
    ):
        # this is an orbax checkpoint
        assert wandb_run_name is not None
        # load config from wandb
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = ml_collections.ConfigDict(run.config)

        # load params
        params = orbax.checkpoint.PyTreeCheckpointer().restore(path, item=None)
        assert "params_ema" not in params

        # load model
        model_def = create_model_def(config.model)
    else:
        # assume this is in HuggingFace format
        model_def, params = load_pretrained_unet(path, in_channels=8)

        # hardcode scheduling config to be "scaled_linear" (used by Stable Diffusion)
        config = {"scheduling": {"noise_schedule": "scaled_linear"}}

    state = EmaTrainState(
        step=0,
        apply_fn=model_def.apply,
        params=None,
        params_ema=params,
        tx=None,
        opt_state=None,
    )
    del params

    # load encoders
    vae_encode, vae_decode = load_vae(pretrained_path)
    tokenize, untokenize, text_encode, action_tokenizer = load_text_encoder(
        pretrained_path)
    uncond_prompt_embed = text_encode(tokenize([""]))  # (1, 77, 768)

    log_snr_fn = scheduling.create_log_snr_fn(config["scheduling"])
    sample_loop = partial(sampling.sample_loop, log_snr_fn=log_snr_fn)

    rng = jax.random.PRNGKey(int(time.time()))

    def action_to_tokenid(actions):
        token_ids = action_tokenizer.__call__([actions])
        prompt_template = np.array(
            [49406, 768, 1311, 585, 1012, 789, 953, 2019, 518, 1816])

        final_token_ids = []

        for ids in token_ids:
            temp_ids = np.concatenate([prompt_template, ids, np.array([286])])
            padding = 77 - len(temp_ids)
            token_ids = np.pad(temp_ids, (0, padding), constant_values=49407)
            final_token_ids.append(token_ids)

        return np.array(final_token_ids)

    def sample(image, action, prompt_w=prompt_w, context_w=context_w):
        nonlocal rng

        image = image / 127.5 - 1.0
        image = image[None]
        assert image.shape == (1, 256, 256, 3)

        prompt_embeds = text_encode(action_to_tokenid(action))

        # encode stuff
        rng, encode_rng = jax.random.split(rng)
        contexts = vae_encode(encode_rng, image, scale=False)

        rng, sample_rng = jax.random.split(rng)
        samples = sample_loop(
            sample_rng,
            state,
            contexts,
            prompt_embeds,
            num_timesteps=num_timesteps,
            prompt_w=prompt_w,
            context_w=context_w,
            eta=eta,
            uncond_y=jnp.zeros_like(contexts),
            uncond_prompt_embeds=uncond_prompt_embed,
        )
        samples = vae_decode(samples)
        samples = jnp.clip(jnp.round(samples * 127.5 + 127.5),
                           0, 255).astype(jnp.uint8)

        return jax.device_get(samples[0])

    # def sample(image, prompt, prompt_w=prompt_w, context_w=context_w):
    #     nonlocal rng

    #     image = image / 127.5 - 1.0
    #     image = image[None]
    #     assert image.shape == (1, 256, 256, 3)

    #     prompt_embeds = text_encode(tokenize([prompt]))

    #     # encode stuff
    #     rng, encode_rng = jax.random.split(rng)
    #     contexts = vae_encode(encode_rng, image, scale=False)

    #     rng, sample_rng = jax.random.split(rng)
    #     samples = sample_loop(
    #         sample_rng,
    #         state,
    #         contexts,
    #         prompt_embeds,
    #         num_timesteps=num_timesteps,
    #         prompt_w=prompt_w,
    #         context_w=context_w,
    #         eta=eta,
    #         uncond_y=jnp.zeros_like(contexts),
    #         uncond_prompt_embeds=uncond_prompt_embed,
    #     )
    #     samples = vae_decode(samples)
    #     samples = jnp.clip(jnp.round(samples * 127.5 + 127.5),
    #                        0, 255).astype(jnp.uint8)

    #     return jax.device_get(samples[0])

    return sample
