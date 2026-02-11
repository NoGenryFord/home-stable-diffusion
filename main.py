import src.generate as generate
import src.fast_real_esrgan_upscale as freu


def check_output_dirs() -> None:
    """Ensure output folders exist for base generation and upscaled results."""
    import os

    base_gen_dir = "./output/base_gen/"
    upscaled_dir = "./output/upscaled/"

    os.makedirs(base_gen_dir, exist_ok=True)
    os.makedirs(upscaled_dir, exist_ok=True)
    print(f"\u2713 Output directories checked/created: {base_gen_dir}, {upscaled_dir}")


def app() -> None:
    """Run the CLI flow: prompt input, generate base image, then upscale and save."""
    check_output_dirs()

    user_prompt = input("Enter your prompt (or press Enter for default): ")
    # If the user presses Enter (empty string), treat as None so generate_image will use its default prompt
    if user_prompt is not None and user_prompt.strip() == "":
        user_prompt = None

    generate.generate_image(user_prompt)
    try:
        res = freu.fast_real_esrgan_upscale(
            "./output/base_gen/output.png", "./models/RealESRGAN_x4.pth"
        )
    except Exception as e:
        print(f"Upscaler failed: {e}")
        return

    # Ensure we got a PIL Image back before saving
    try:
        res.save("./output/upscaled/output_upscaled.png")
    except Exception as e:
        print(f"Failed to save upscaled image: {e}")
        return

    print("All done! Upscaled image saved to ./output/upscaled/output_upscaled.png")


if __name__ == "__main__":
    app()
