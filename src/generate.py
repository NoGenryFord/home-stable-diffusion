from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
import torch
from PIL import Image
from typing import Optional


# Main function to generate image
def generate_image(user_prompt: Optional[str] = None, save_dir: str = "./output/base_gen/") -> None:
    """Generate a base image with Stable Diffusion and save it to disk.

    Args:
        user_prompt: Prompt text for generation. If None, a default prompt is used.
        save_dir: Output directory where the base image is saved.
    """
    # cheick if avaible MPS or CUDA
    if torch.backends.mps.is_available():
        device = "mps"
        print("\u2713 using MPS (Metal) (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("\u2713 using CUDA (NVIDIA GPU)")
    else:
        device = "cpu"
        print("\u26a0 Using CPU")

    # Loading model
    print("Loading model...")
    txt2img = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Or "stabilityai/stable-diffusion-2-1"
        torch_dtype=torch.float32,
        safety_checker=None,  # Disabling safety checker for speed
    )
    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Or "stabilityai/stable-diffusion-2-1"
        torch_dtype=torch.float32,
        safety_checker=None,  # Disabling safety checker for speed
    )

    # Loading better VAE for improved image quality
    better_vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    )
    better_vae = better_vae.to(device, dtype=torch.float32)
    better_vae.eval()
    better_vae.requires_grad_(False)
    better_vae.config.force_upcast = True

    # Moving VAE to device with float32 for better quality
    txt2img.vae = better_vae

    txt2img = txt2img.to(device)
    txt2img.enable_attention_slicing()
    print("\u2713 model txt2img loaded")
    # For img2img
    img2img.vae = better_vae

    img2img = img2img.to(device)
    img2img.enable_attention_slicing()
    print("\u2713 model img2img loaded")

    # Setting scheduler to DPM++ 2M Karras
    txt2img.scheduler = DPMSolverMultistepScheduler.from_config(
        txt2img.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    img2img.scheduler = txt2img.scheduler
    print("\u2713 scheduler: DPM++ 2M Karras")

    # Generating image
    prompt = user_prompt
    negative_prompt = (
        "low quality, worst quality, blurry, out of focus, "
        "jpeg artifacts, compression artifacts, noise, grain, "
        "overexposed, underexposed, oversaturated, "
        "deformed, distorted, disfigured, mutation, ugly, "
        "extra limbs, extra fingers, fused fingers, "
        "collage, mosaic, watermark, text, logo"
    )

    if prompt is None:
        prompt = "A high-resolution photo of a beautiful landscape, vibrant colors, detailed, professional photography"
    print(f"Generating image...")

    # --- txt2img at 640x640 -> latent ---
    result = txt2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=30,  # Number of steps (more = higher quality but slower)
        guidance_scale=5.5,  # How strongly to follow the prompt
    )

    image = result.images[0]
    print("\u2713 generation complete at 512x512")

    # --- img2img hi-res fix ---
    hires_size = (1024, 1024)  # Final size after hi-res fix
    # Use Pillow's Resampling enum for compatibility with newer versions
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # Fallback for older Pillow versions
        resample_filter = Image.LANCZOS

    image_upscaled = image.resize(hires_size, resample=resample_filter)
    print("Performing img2img hi-res fix...")
    result = img2img(
        prompt=prompt,
        image=image_upscaled,
        num_inference_steps=30,  # Number of steps (more = higher quality but slower)
        guidance_scale=5.5,  # How strongly to follow the prompt
        strength=0.35,  # Denoising strength
        output_type="pil",
    )

    # Getting image from result
    image = result.images[0]
    print("\u2713 img2img denoise complete")

    # Saving
    image.save(f"{save_dir}output.png")
    print(f"\u2713 saved to {save_dir}output.png")
