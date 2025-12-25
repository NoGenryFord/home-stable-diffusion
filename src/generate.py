from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
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

    # Moving VAE to device with float32 for better quality
    txt2img.vae = txt2img.vae.to(device, dtype=torch.float32)
    txt2img.vae.eval()
    txt2img.vae.requires_grad_(False)
    txt2img.vae.config.force_upcast = True

    txt2img = txt2img.to(device)
    txt2img.enable_attention_slicing()
    print("✓ model txt2img loaded")
    # For img2img
    img2img.vae = txt2img.vae

    img2img = img2img.to(device)
    img2img.enable_attention_slicing()
    print("✓ model img2img loaded")

    # Generating image

    prompt = user_prompt
    negative_prompt = (
        "low quality, worst quality, jpeg artifacts, "
        "deformed, distorted, disfigured, mutation, "
        "extra limbs, extra objects, collage, mosaic"
    )

    if prompt is None:
        prompt = "A high-resolution photo of a beautiful landscape, vibrant colors, detailed, professional photography"
    print(f"Generating image...")

    # --- txt2img at 640x640 -> latent ---
    result = txt2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=640,
        width=640,
        num_inference_steps=25,  # Number of steps (more = higher quality but slower)
        guidance_scale=6.0,  # How strongly to follow the prompt
    )

    image = result.images[0]
    print("✓ generation complete at 640x640")

    # --- img2img denoise ---
    print("Performing img2img denoise...")
    result = img2img(
        prompt=prompt,
        image=image,
        num_inference_steps=50,  # Number of steps (more = higher quality but slower)
        guidance_scale=6.0,  # How strongly to follow the prompt
        strength=0.25,  # Denoising strength
        output_type="pil",
    )

    # Getting image from result
    image = result.images[0]
    print("✓ img2img denoise complete")

    # Saving
    image.save(f"{save_dir}output.png")
    print(f"✓ saved to {save_dir}output.png")
