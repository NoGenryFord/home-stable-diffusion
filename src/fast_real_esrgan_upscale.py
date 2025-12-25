import torch
from PIL import Image
import numpy as np
from spandrel import ModelLoader


def fast_real_esrgan_upscale(img_input, model_path):
    # 1. Loading model via Spandrel
    loader = ModelLoader()
    model = loader.load_from_file(model_path)

    # Determining device (NVIDIA GPU or MPS or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    # 2. Preparing image (PIL -> Tensor)
    # If img_input is a file path, open it
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input.convert("RGB")

    img_array = np.array(img).astype(np.float32) / 255.0
    # Change. to format [B, C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

    # 3. Performing upscale
    with torch.no_grad():
        try:
            output_tensor = model(img_tensor)
        except RuntimeError as e:
            return f"Memory error: {e}. Try x2 model or smaller image."

    # 4. Returning to PIL format
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    output_img = (output_img * 255).astype(np.uint8)
    return Image.fromarray(output_img)
