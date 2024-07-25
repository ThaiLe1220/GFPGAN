import os
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
import random
from io import BytesIO
import concurrent.futures
from tqdm import tqdm


def add_gaussian_noise(image, noise_factor=0.1):
    img_array = np.array(image)
    noise = np.random.normal(0, 255 * noise_factor, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def add_salt_pepper_noise(image, salt_vs_pepper=0.3, amount=0.04):
    img_array = np.array(image)
    num_salt = np.ceil(amount * img_array.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img_array.size * (1.0 - salt_vs_pepper))

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 255

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 0

    return Image.fromarray(img_array)


def simulate_multiple_jpeg_compressions(image, min_compressions=1, max_compressions=5):
    num_compressions = random.randint(min_compressions, max_compressions)

    for _ in range(num_compressions):
        quality = random.randint(30, 95)
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer)

    return image


def adjust_white_balance(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def process_image(input_path, output_path, options):
    img = Image.open(input_path)

    # Random crop
    if random.random() < options["crop_probability"]:
        width, height = img.size
        crop_size = int(min(width, height) * random.uniform(0.7, 1.0))
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        img = img.crop((left, top, left + crop_size, top + crop_size))

    # Resize
    img_processed = img.resize((512, 512), Image.LANCZOS)

    # Blur
    if random.random() < options["blur_probability"]:
        blur_type = random.choice(["gaussian", "median"])
        blur_strength = random.randint(1, 8)
        if blur_type == "gaussian":
            img_processed = img_processed.filter(
                ImageFilter.GaussianBlur(blur_strength)
            )
        else:
            img_cv = cv2.cvtColor(np.array(img_processed), cv2.COLOR_RGB2BGR)
            img_processed = cv2.medianBlur(img_cv, blur_strength * 2 + 1)
            img_processed = Image.fromarray(
                cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            )

    # Noise
    if random.random() < options["noise_probability"]:
        noise_type = random.choice(["gaussian", "salt_pepper"])
        if noise_type == "gaussian":
            img_processed = add_gaussian_noise(img_processed, random.uniform(0.01, 0.1))
        else:
            img_processed = add_salt_pepper_noise(
                img_processed, random.uniform(0.3, 0.7), random.uniform(0.005, 0.02)
            )

    # Color degradation
    if random.random() < options["color_degradation_probability"]:
        img_processed = img_processed.quantize(colors=random.randint(8, 64)).convert(
            "RGB"
        )

    # White balance adjustment
    if random.random() < options["white_balance_probability"]:
        img_processed = adjust_white_balance(img_processed, random.uniform(0.5, 1))

    # Multiple compression (simulate multiple saves)
    if random.random() < options["multiple_compression_probability"]:
        img_processed = simulate_multiple_jpeg_compressions(img_processed, 2, 5)

    img_processed.save(output_path, quality=95)


def create_degraded_images(input_folder, output_folder, options):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"degraded_{filename}")
            process_image(input_path, output_path, options)
            print(f"Processed: {filename}")


def process_image_wrapper(args):
    input_path, output_path, options = args
    if os.path.exists(output_path):
        return "skipped"
    try:
        process_image(input_path, output_path, options)
        return "processed"
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return "error"


def process_directory(input_dir, output_dir, options, num_threads=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    args_list = []
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_filename = os.path.splitext(img_file)[0] + ".jpg"
        output_path = os.path.join(output_dir, output_filename)
        args_list.append((input_path, output_path, options))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(
            tqdm(
                executor.map(process_image_wrapper, args_list),
                total=len(args_list),
                desc="Processing Images",
            )
        )

    processed = sum(1 for r in results if r == "processed")
    skipped = sum(1 for r in results if r == "skipped")
    errors = sum(1 for r in results if r == "error")

    print(
        f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}, Total: {len(image_files)}"
    )


if __name__ == "__main__":
    input_directory = "/home/ubuntu/Desktop/eugene/GFPGAN/datasets/ffhq/1k"
    output_directory = "/home/ubuntu/Desktop/eugene/GFPGAN/datasets/ffhq/1k_lq"

    opts = {
        "crop_probability": 0.05,
        "blur_probability": 0.3,
        "noise_probability": 0.1,
        "color_degradation_probability": 0.3,
        "white_balance_probability": 0.4,
        "multiple_compression_probability": 0.7,
    }

    process_directory(input_directory, output_directory, opts, num_threads=12)

    # process_image(
    #     "/home/ubuntu/Desktop/eugene/GFPGAN/datasets/test1.png",
    #     "/home/ubuntu/Desktop/eugene/GFPGAN/datasets/lq_test1.jpg",
    #     options,
    # )
