from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
import torch
import torch.nn.functional as F
from PIL import Image


# Main function to generate image
def generate_image(user_prompt: str = None, save_dir: str = "./output/base_gen/"):
    # cheick if avaible MPS or CUDA
    if torch.backends.mps.is_available():
        device = "mps"
        print("✓ using MPS (Metal) (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✓ using CUDA (NVIDIA GPU)")
    else:
        device = "cpu"
        print("⚠ Using CPU")

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
    print("✓ model txt2img loaded")
    # For img2img
    img2img.vae = better_vae

    img2img = img2img.to(device)
    img2img.enable_attention_slicing()
    print("✓ model img2img loaded")

    # Setting scheduler to DPM++ 2M Karras
    txt2img.scheduler = DPMSolverMultistepScheduler.from_config(
        txt2img.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    img2img.scheduler = txt2img.scheduler
    print("✓ scheduler: DPM++ 2M Karras")

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
    print("✓ generation complete at 512x512")

    # --- img2img hi-res fix ---
    hires_size = (768, 768)  # Final size after hi-res fix
    image_upscaled = image.resize(hires_size, resample=Image.LANCZOS)
    print("Performing img2img hi-res fix...")
    result = img2img(
        prompt=prompt,
        image=image,
        num_inference_steps=25,  # Number of steps (more = higher quality but slower)
        guidance_scale=5.5,  # How strongly to follow the prompt
        strength=0.35,  # Denoising strength
        output_type="pil",
    )

    # Getting image from result
    image = result.images[0]
    print("✓ img2img denoise complete")

    # Saving
    image.save(f"{save_dir}output.png")
    print(f"✓ saved to {save_dir}output.png")
