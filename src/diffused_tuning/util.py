import torch
from diffusers import DiffusionPipeline
from PIL import Image
import io

def parse_img_from_hex(hex_str: str) -> Image.Image:
    img_bytes = bytes.fromhex(hex_str)
    return Image.open(io.BytesIO(img_bytes))

def print_img_as_hex(img: Image.Image) -> None:
    with io.BytesIO() as f:
        img.save(f, format="PNG")
        img_bytes = f.getvalue()
    print(f"IMAGE={img_bytes.hex()}")


def pipeline_callback(step: int, timestep: int, latents: torch.FloatTensor, *, pipe: DiffusionPipeline) -> None:
    img = pipe.numpy_to_pil(pipe.decode_latents(latents))[0]
    print(f"PROGRESS={step}")
    print_img_as_hex(img)