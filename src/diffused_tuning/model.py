import argparse
import functools
import diffused_tuning.util as util

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline

def main():
    args = parse_args()
    model_func = {
        "inpaint": inpaint_image_from_cli,
        "generate": generate_image_from_cli,
    }[args.model_version]

    model_func(args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=str, required=True, choices=["inpaint", "generate"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--inpaint-mask-hex", type=str, required=False)
    return parser.parse_args()


def generate_image_from_cli(args):
    """
    Prints the generated image as a hex string to stdout at each step.
    Also saves the final image to img.png.

    Example usage:
    python model.py --model-version generate --prompt="A cat in a hat" --size=768 --num_steps=20 --guidance=7.5
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance,
        num_images_per_prompt=1,
        callback=functools.partial(util.pipeline_callback, pipe=pipe),
    ).images

    img = images[0]
    util.print_img_as_hex(img)
    img.save("img.png")

def inpaint_image_from_cli(args):
    """
    Prints the generated image as a hex string to stdout at each step.
    Also saves the final image to img.png.

    CLI usage not suitable, use the GUI instead.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    mask = util.parse_img_from_hex(args.inpaint_mask_hex)
    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance,
        num_images_per_prompt=1,
        callback=functools.partial(util.pipeline_callback, pipe=pipe),
        mask_image=mask,
    ).images

    img = images[0]
    util.print_img_as_hex(img)
    img.save("img.png")

if __name__ == "__main__":
    main()