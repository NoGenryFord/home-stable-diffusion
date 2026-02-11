import torch
from PIL import Image
import numpy as np
from spandrel import ModelLoader
from typing import Union


def fast_real_esrgan_upscale(img_input: Union[str, Image.Image], model_path: str) -> Image.Image:
    """Upscale an image with a RealESRGAN-compatible model via Spandrel.

    Args:
        img_input: Input image or file path to an image.
        model_path: Path to a RealESRGAN model file (.pth).

    Returns:
        A PIL Image with the upscaled result.

    Raises:
        RuntimeError: If the model fails due to memory issues.
        TypeError: If the model output type is unexpected.
    """
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
            output = model(img_tensor)
        except RuntimeError as e:
            # Raise a proper exception so caller can handle it
            raise RuntimeError(f"Memory error during upscaling: {e}. Try a smaller image or a x2 model.")

    # Model may return tensor directly or a tuple/dict
    if isinstance(output, tuple) or isinstance(output, list):
        output_tensor = output[0]
    elif isinstance(output, dict):
        # try common keys
        for k in ("output", "pred", "image", "out"):
            if k in output:
                output_tensor = output[k]
                break
        else:
            # fall back to first value
            output_tensor = next(iter(output.values()))
    else:
        output_tensor = output

    # Ensure tensor is on CPU and in [B, C, H, W]
    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError(f"Model returned unexpected type: {type(output_tensor)}")

    output_tensor = output_tensor.detach().cpu()

    # If output is in [-1,1] range, convert; if in [0,1], keep
    # Heuristics: check max/min
    tmin = float(output_tensor.min())
    tmax = float(output_tensor.max())
    if tmin >= -1.1 and tmax <= 1.1:
        # assume [-1,1]
        output_tensor = (output_tensor + 1.0) / 2.0
    # else assume already [0,1]

    # Remove batch dim and move channels to last
    output_tensor = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    output_img = (output_tensor * 255).astype(np.uint8)
    return Image.fromarray(output_img)
