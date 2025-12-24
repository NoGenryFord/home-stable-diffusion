import src.generate as generate
import src.latend_diff_sd_upscaler as latend_diff_sd_upscaler


def app():
    user_prompt = input("Enter your prompt (or press Enter for default): ")

    generate.generate_image(user_prompt)
    res = latend_diff_sd_upscaler.fast_real_esrgan_upscale(
        "./output/base_gen/output.png", "./models/RealESRGAN_x4.pth"
    )
    res.save("./output/upscaled/output_upscaled.png")
    print("All done! Upscaled image saved to ./output/upscaled/output_upscaled.png")


if __name__ == "__main__":
    app()
