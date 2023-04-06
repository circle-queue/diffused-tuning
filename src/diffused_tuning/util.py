import io
import base64


import torch
from PIL import Image


def hex_to_img(hex_str: str) -> Image.Image:
    img_bytes = bytes.fromhex(hex_str)
    return Image.open(io.BytesIO(img_bytes))


def img_to_hex(img: Image.Image) -> str:
    with io.BytesIO() as f:
        img.save(f, format="PNG")
        return f.getvalue().hex()


def img_to_dataurl(img: Image.Image) -> str:
    hex_str = img_to_hex(img)
    byte_str = bytes.fromhex(hex_str)
    base64_str = base64.b64encode(byte_str).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def dataurl_to_img(url: str) -> Image.Image:
    _, base64_str = url.split("base64,")
    byte_str = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(byte_str))


def pipeline_callback(step: int, timestep: int, latents: torch.FloatTensor, *, pipe) -> None:
    img = pipe.numpy_to_pil(pipe.decode_latents(latents))[0]
    print(f"PROGRESS={step}")
    print(f"IMAGE={img_to_hex(img)}")
