import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

size = (512, 512)
batch_image_size = (1, 512, 512, 3)
batch_mask_size = (1, 512, 512, 1)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "A dwarf standing in a fantasy landscape"

image = Image.open("img.png").resize(size)

# The mask structure is white for inpainting and black for keeping as is
mask_image = Image.open("mask.png").convert("L").resize(size)

image_tensor = torch.tensor(np.array(image)[..., :3], dtype=torch.float16).reshape(
    batch_image_size
)
mask_tensor = torch.tensor(np.array(mask_image), dtype=torch.float16).reshape(
    batch_mask_size
)

images = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_images_per_prompt=2,
    num_inference_steps=50,
    guidance_scale=7,
).images
display(*images)
