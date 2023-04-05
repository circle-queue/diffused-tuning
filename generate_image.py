import argparse
from PIL import Image
import io

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
)
pipe.to("cuda")


def print_img_as_hex(img: Image):
    with io.BytesIO() as f:
        img.save(f, format="PNG")
        img_bytes = f.getvalue()
    print(f"IMAGE={img_bytes.hex()}")


def callaback(step: int, timestep: int, latents: torch.FloatTensor):
    img = pipe.numpy_to_pil(pipe.decode_latents(latents))[0]
    print(f"PROGRESS={step}")
    print_img_as_hex(img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    return parser.parse_args()


def generate_image_from_cli():
    """
    Prints the generated image as a hex string to stdout at each step.
    Also saves the final image to img.png.

    Example usage:
    python generate_image.py --prompt="A cat in a hat" --size=768 --num_steps=20 --guidance=7.5
    """
    args = parse_args()
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance,
        num_images_per_prompt=1,
        callback=callaback,
    ).images

    img = images[0]
    print_img_as_hex(img)
    img.save("img.png")


if __name__ == "__main__":
    generate_image_from_cli()
