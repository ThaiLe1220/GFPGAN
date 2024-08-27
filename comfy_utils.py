import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
import time


# Global variables
CUSTOM_PATH = False
NODE_CLASS_MAPPINGS = None
TILEPREPROCESSOR = None
LINEARTPREPROCESSOR = None
EFFICIENT_LOADER = None
VAEENCODE = None
KSAMPLER_EFFICIENT = None


def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        return os.path.join(path, name)

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    global CUSTOM_PATH
    if not CUSTOM_PATH:
        comfyui_path = find_path("ComfyUI")
        if comfyui_path is not None and os.path.isdir(comfyui_path):
            sys.path.append(comfyui_path)
            print(f"'{comfyui_path}' added to sys.path")
            CUSTOM_PATH = True
        else:
            print("ComfyUI directory not found")


add_comfyui_directory_to_sys_path()

import comfy.controlnet
import folder_paths


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    print(comfyui_path)
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")
        print(f"'{comfyui_path}' added to sys.path")


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    init_builtin_extra_nodes()
    init_external_custom_nodes()


def add_extra_model_paths() -> None:
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def setup_comfy_environment():
    global NODE_CLASS_MAPPINGS
    global TILEPREPROCESSOR, LINEARTPREPROCESSOR
    global EFFICIENT_LOADER, VAEENCODE, KSAMPLER_EFFICIENT

    import_custom_nodes()

    from nodes import NODE_CLASS_MAPPINGS

    # TILEPREPROCESSOR = NODE_CLASS_MAPPINGS["TilePreprocessor"]()
    LINEARTPREPROCESSOR = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
    EFFICIENT_LOADER = NODE_CLASS_MAPPINGS["Efficient Loader"]()
    VAEENCODE = NODE_CLASS_MAPPINGS["VAEEncode"]()
    KSAMPLER_EFFICIENT = NODE_CLASS_MAPPINGS["KSampler (Efficient)"]()


PROMPT_DATA = json.loads(
    "{"
    '"12": {"inputs": {"seed": 999999999,"steps": 4,"cfg": 1.5,"sampler_name": "dpmpp_sde","scheduler": "karras","denoise": 0.35,"preview_method": "auto","vae_decode": "true","model": ["14",0],"positive": ["14",1],"negative": ["14",2],"latent_image": ["13",0],"optional_vae": ["14",4]},"class_type": "KSampler (Efficient)","_meta": {"title": "KSampler (Efficient)"}},'
    '"13": {"inputs": {"pixels": "image_tensor","vae": ["14",4]},"class_type": "VAEEncode","_meta": {"title": "VAE Encode"}},'
    '"14": {"inputs": {"ckpt_name": "realisticVisionV60B1_v51VAE.safetensors","vae_name": "vae-ft-mse-840000-ema-pruned.safetensors","clip_skip": -1,"lora_name": "add_detail.safetensors","lora_model_strength": 0.5,"lora_clip_strength": 0.5,"positive":"positive_prompt","negative": "negative_prompt","token_normalization": "none","weight_interpretation": "comfy","empty_latent_width": 512,"empty_latent_height": 512,"batch_size": 1,"cnet_stack": "controlnet_list"},"class_type": "Efficient Loader","_meta": {"title": "Efficient Loader"}},'
    '"281": {"inputs": {"pyrUp_iters": 1,"resolution": 512,"image": "image_tensor"},"class_type": "TilePreprocessor","_meta": {"title": "Tile"}},'
    '"282": {"inputs": {"coarse": "disable","resolution": 512,"image": "image_tensor"},"class_type": "LineArtPreprocessor","_meta": {"title": "Realistic Lineart"}}'
    "}"
)


def face_enhance_inference(
    input_face_tensor,
    processed_face_tensor,
    lora_strength=0.5,
    cnet_strength=1.0,
    ksampler_steps=4,
    ksampler_seed=random.randint(1, 2**64),
    ksampler_denoise=0.35,
    positive_prompt="masterpiece,best quality,(photorealistic:1.2),8k raw photo,bokeh,beautifully detailed face,clear eyes,smooth skin texture,depth of field,perfect dental arch,clear skin pores",
    negative_prompt="(nsfw,naked,nude,deformed iris,deformed pupils,semi-realistic,cgi,3d,render,sketch,cartoon,anime),(deformed,distorted,disfigured:1.3),poorly drawn,mutated,ugly,disgusting,amputation,drawing,paiting,crayon,sketch,graphite,impressionist,noisy,blurry,soft,deformed,ugly,lowers,bad anatomy,text,error,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username",
):
    with torch.inference_mode():
        lineartpreprocessor_282 = LINEARTPREPROCESSOR.execute(
            coarse="disable",
            resolution=512,
            image=preprocess_tensor(processed_face_tensor),
        )

        def create_controlnet_list(
            controlnet_1,
            controlnet_strength_1,
            image_1,
            start_percent_1,
            end_percent_1,
        ):

            controlnet_list = []

            # Helper function to load and add ControlNet to the list
            def add_controlnet(name, strength, image, start_percent, end_percent):
                if name != "None" and image is not None:
                    controlnet_path = folder_paths.get_full_path("controlnet", name)
                    controlnet = comfy.controlnet.load_controlnet(controlnet_path)
                    controlnet_list.append(
                        (controlnet, image, strength, start_percent, end_percent)
                    )

            # # Add ControlNet 1
            add_controlnet(
                controlnet_1,
                controlnet_strength_1,
                image_1,
                start_percent_1,
                end_percent_1,
            )

            return controlnet_list

        controlnet_list = create_controlnet_list(
            "control_v11p_sd15_lineart_fp16.safetensors",
            cnet_strength,
            get_value_at_index(lineartpreprocessor_282, 0),
            0.0,
            1.0,
        )

        efficient_loader_14 = EFFICIENT_LOADER.efficientloader(
            ckpt_name="realisticVisionV60B1_v51HyperVAE.safetensors",
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors",
            clip_skip=-1,
            lora_name="add_detail.safetensors",
            lora_model_strength=lora_strength,
            lora_clip_strength=lora_strength,
            positive=positive_prompt,
            negative=negative_prompt,
            token_normalization="none",
            weight_interpretation="comfy",
            empty_latent_width=512,
            empty_latent_height=512,
            batch_size=1,
            cnet_stack=controlnet_list,
            prompt=PROMPT_DATA,
        )

        vaeencode_13 = VAEENCODE.encode(
            pixels=img2tensor_to_load_image(processed_face_tensor),
            vae=get_value_at_index(efficient_loader_14, 4),
        )

        ksampler_efficient_12 = KSAMPLER_EFFICIENT.sample(
            seed=ksampler_seed,
            steps=ksampler_steps,
            cfg=1.5,
            sampler_name="dpmpp_sde",
            scheduler="karras",
            denoise=ksampler_denoise,
            preview_method="auto",
            vae_decode="true",
            model=get_value_at_index(efficient_loader_14, 0),
            positive=get_value_at_index(efficient_loader_14, 1),
            negative=get_value_at_index(efficient_loader_14, 2),
            latent_image=get_value_at_index(vaeencode_13, 0),
            optional_vae=get_value_at_index(efficient_loader_14, 4),
            prompt=PROMPT_DATA,
        )

        return img2tensor_to_load_image(get_value_at_index(ksampler_efficient_12, 5))


def save_image(image, output_filename, output_folder="./"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert the PyTorch tensor to a numpy array
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = image

    # Ensure the image is in the correct format (H, W, C) and scale to 0-255
    if image_np.ndim == 4:
        image_np = image_np.squeeze(0)  # Remove batch dimension if present
    if image_np.shape[0] == 3:
        image_np = np.transpose(
            image_np, (1, 2, 0)
        )  # Change from (C, H, W) to (H, W, C)

    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    # Create a PIL Image
    img = Image.fromarray(image_np)

    # Save the image
    output_path = os.path.join(output_folder, output_filename)
    img.save(output_path, format="PNG")
    print(f"Saved image to {output_path}")
