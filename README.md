# home-stable-diffusion

[English](#english) | [Українська](#українська)

---

## English

Image generation via Stable Diffusion with RealESRGAN upscaling.

### What it does

1. Generates images from text prompts (txt2img)
2. Upscales resolution via RealESRGAN x4
3. Saves result to `output/upscaled/`

### Installation

#### Dependencies

```bash
uv sync
```

#### Models

Required models download automatically on first run:
- `runwayml/stable-diffusion-v1-5` - base SD model
- `stabilityai/sd-vae-ft-mse` - improved VAE

Download RealESRGAN manually:
```bash
mkdir -p models
cd models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O RealESRGAN_x4.pth
```

### Usage

```bash
uv run python main.py
```

Enter prompt or press Enter for default.

### Structure

```
main.py                    # Entry point
src/generate.py            # SD generation
src/fast_real_esrgan_upscale.py  # Upscaling
models/RealESRGAN_x4.pth   # Upscaler model
output/base_gen/           # Base generation
output/upscaled/           # Final result
```

### GPU Support

- CUDA (NVIDIA)
- MPS (Apple Silicon)
- CPU (fallback)

Automatic device detection.

---

## Українська

Генерація зображень через Stable Diffusion з подальшим апскейлінгом через RealESRGAN.

### Що робить

1. Генерує зображення з текстового промпта (txt2img)
2. Підвищує роздільність через RealESRGAN x4
3. Зберігає результат в `output/upscaled/`

### Встановлення

#### Залежності

```bash
uv sync
```

#### Моделі

Потрібні моделі завантажуються автоматично при першому запуску:
- `runwayml/stable-diffusion-v1-5` - базова модель SD
- `stabilityai/sd-vae-ft-mse` - покращений VAE

Вручну завантажте RealESRGAN:
```bash
mkdir -p models
cd models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O RealESRGAN_x4.pth
```

### Запуск

```bash
uv run python main.py
```

Введіть промпт або натисніть Enter для дефолтного.

### Структура

```
main.py                    # Точка входу
src/generate.py            # Генерація SD
src/fast_real_esrgan_upscale.py  # Апскейлінг
models/RealESRGAN_x4.pth   # Модель апскейлера
output/base_gen/           # Базова генерація
output/upscaled/           # Фінальний результат
```

### Підтримка GPU

- CUDA (NVIDIA)
- MPS (Apple Silicon)
- CPU (fallback)

Автоматичне визначення доступного пристрою.
