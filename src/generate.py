from diffusers import StableDiffusionPipeline
import torch


def generate_image(user_prompt: str = None):
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
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Or "stabilityai/stable-diffusion-2-1"
        torch_dtype=torch.float32,
        safety_checker=None,  # Disabling safety checker for speed
    )

    pipe.vae = pipe.vae.to(device, dtype=torch.float32)
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)
    pipe.vae.config.force_upcast = True

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # Generating image

    prompt = user_prompt

    if prompt is None:
        prompt = "A high-resolution photo of a beautiful landscape, vibrant colors, detailed, professional photography"
    print(f"Generating image: {prompt}")

    image = pipe(
        prompt,
        num_inference_steps=20,  # Number of steps (more = higher quality but slower)
        guidance_scale=7.0,  # How strongly to follow the prompt
        height=640,
        width=640,
    ).images[0]

    # Saving
    image.save("./output/base_gen/output.png")
    print("✓ saved to output.png")
    print("Done!")
