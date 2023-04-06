import io
import base64
import gzip


import torch
from PIL import Image


def b64_to_img(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(b64_string_to_bytes(b64_str)))


def img_to_b64(img: Image.Image) -> str:
    with io.BytesIO() as f:
        img.save(f, format="PNG")
        return byte_string_to_b64(f.getvalue())


def byte_string_to_b64(byte_str: bytes) -> str:
    return base64.b64encode(byte_str).decode("utf-8")


def b64_string_to_bytes(b64_str) -> bytes:
    return base64.b64decode(b64_str)


def compressed_b64(b64_str) -> str:
    return byte_string_to_b64(gzip.compress(b64_string_to_bytes(b64_str)))


def decompress_b64(b64_str) -> str:
    return byte_string_to_b64(gzip.decompress(b64_string_to_bytes(b64_str)))


def b64_to_dataurl(b64_str: str) -> str:
    return f"data:image/png;base64,{b64_str}"


def dataurl_to_b64(url: str) -> str:
    _, base64_str = url.split("base64,")
    return base64_str


def pipeline_callback(step: int, timestep: int, latents: torch.FloatTensor, *, pipe) -> None:
    img = pipe.numpy_to_pil(pipe.decode_latents(latents))[0]
    print(f"PROGRESS={step}")
    print(f"IMAGE={img_to_b64(img)}")
