from diffusers import StableDiffusionPipeline
import torch

# Перевірка доступності MPS
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ MPS (Metal) доступний")
else:
    device = "cpu"
    print("⚠ Використовується CPU")

# Завантаження моделі
print("Завантаження моделі...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Або "stabilityai/stable-diffusion-2-1"
    torch_dtype=torch.float32,
    safety_checker=None,  # Відключаємо safety checker для швидкості
)

pipe.vae = pipe.vae.to(device, dtype=torch.float32)
pipe.vae.eval()
pipe.vae.requires_grad_(False)
pipe.vae.config.force_upcast = True

pipe = pipe.to(device)
pipe.enable_attention_slicing()

# Генерація
prompt = "A high-detail digital artwork of a futuristic cupola colony on Mars, featuring intricate geodesic dome structures with transparent panels revealing habitation modules inside, rust-colored Martian landscape with rocky terrain and dust, dramatic lighting from the distant sun casting long shadows, cinematic composition, professional concept art style, sharp focus, volumetric atmosphere, highly detailed, photorealistic rendering"
print(f"Генерую зображення: {prompt}")

image = pipe(
    prompt,
    num_inference_steps=20,  # Кількість кроків (більше = якісніше, але повільніше)
    guidance_scale=7.0,  # Наскільки сильно слідувати промпту
    height=640,
    width=640,
).images[0]

# Збереження
image.save("./output/base_gen/output.png")
print("✓ Збережено в output.png")
print("Готово!")
