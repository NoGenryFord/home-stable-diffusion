import torch
from PIL import Image
import numpy as np
from spandrel import ModelLoader


def fast_real_esrgan_upscale(img_input, model_path):
    # 1. Завантаження моделі через Spandrel
    loader = ModelLoader()
    model = loader.load_from_file(model_path)

    # Визначаємо девайс (NVIDIA GPU або CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Підготовка зображення (PIL -> Tensor)
    # Якщо img_input - це шлях до файлу, відкриваємо його
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input.convert("RGB")

    img_array = np.array(img).astype(np.float32) / 255.0
    # Перетворюємо в формат [B, C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

    # 3. Виконання апскейлу
    with torch.no_grad():
        try:
            output_tensor = model(img_tensor)
        except RuntimeError as e:
            return f"Помилка пам'яті: {e}. Спробуй модель x2 або зменш картинку."

    # 4. Повернення в PIL формат
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    output_img = (output_img * 255).astype(np.uint8)
    return Image.fromarray(output_img)


# Приклад використання:
# res = fast_real_esrgan_upscale("input_640.png", "RealESRGAN_x4plus.pth")
# res.save("output_2560.png")

res = fast_real_esrgan_upscale(
    "./output/base_gen/output.png", "./models/RealESRGAN_x4.pth"
)
res.save("./output/upscaled/outpute_upscaled.png")
