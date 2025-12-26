import src.generate as generate
import src.fast_real_esrgan_upscale as freu


def check_output_dirs():
    import os

    base_gen_dir = "./output/base_gen/"
    upscaled_dir = "./output/upscaled/"

    os.makedirs(base_gen_dir, exist_ok=True)
    os.makedirs(upscaled_dir, exist_ok=True)
    print(f"✓ Output directories checked/created: {base_gen_dir}, {upscaled_dir}")


def app():
    check_output_dirs()

    user_prompt = input("Enter your prompt (or press Enter for default): ")

    generate.generate_image(user_prompt)
    res = freu.fast_real_esrgan_upscale(
        "./output/base_gen/output.png", "./models/RealESRGAN_x4.pth"
    )
    res.save("./output/upscaled/output_upscaled.png")
    print("All done! Upscaled image saved to ./output/upscaled/output_upscaled.png")


if __name__ == "__main__":
    app()
