import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def merge_and_compare_images(input_dir, output_dir, *args):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parse input directories and names
    dirs = [input_dir] + [arg for i, arg in enumerate(args) if i % 2 == 0]
    names = ["input_plus"] + [arg for i, arg in enumerate(args) if i % 2 == 1]

    def add_version_text(img, text, font_size):
        new_img = Image.new(
            "RGB", (img.width, img.height + int(font_size * 1.2)), color="white"
        )
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        font = ImageFont.load_default(font_size)
        text_width = draw.textlength(text, font=font)
        draw.text(
            ((img.width - text_width) / 2, img.height), text, fill="black", font=font
        )
        return new_img

    def find_file_with_base_name(directory, base_name):
        for ext in [".png", ".jpeg", ".jpg"]:
            if os.path.exists(os.path.join(directory, base_name + ext)):
                return os.path.join(directory, base_name + ext)
        return None

    def merge_images(file):
        base_name = os.path.splitext(file)[0]
        paths = [find_file_with_base_name(dir, base_name) for dir in dirs]

        if not all(paths):
            return f"Error: {base_name} not found in one of the directories"

        images = [Image.open(path) for path in paths]

        font_size = max(images[0].width, images[0].height) / 18
        images_with_text = [
            add_version_text(img, f"{name} ({img.width}x{img.height})", font_size)
            for img, name in zip(images, names)
        ]

        max_height = max(img.height for img in images_with_text)
        total_width = sum(img.width for img in images_with_text)

        merged = Image.new(
            "RGB",
            (total_width, max_height),
            color="white",
        )

        x_offset = 0
        for img in images_with_text:
            merged.paste(img, (x_offset, (max_height - img.height) // 2))
            x_offset += img.width

        merged.save(os.path.join(output_dir, base_name + ".png"))
        return f"Merged {base_name}"

    # Get list of files in first directory
    files = [
        f
        for f in os.listdir(dirs[1])  # Use the first processed directory
        if (
            f.lower().startswith(
                ("deformed_", "image", "shorpy", "v1.4", "input", "vRF", "v1024")
            )
            and f.lower().endswith((".png", ".jpeg", ".jpg"))
        )
    ]

    # Use ThreadPoolExecutor for multi-threading with max_workers=12
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_file = {executor.submit(merge_images, file): file for file in files}

        for future in tqdm(
            as_completed(future_to_file), total=len(files), desc="Merging images"
        ):
            file = future_to_file[future]
            try:
                result = future.result()
                if result.startswith("Error"):
                    print(result)
            except Exception as exc:
                print(f"{file} generated an exception: {exc}")

    print("Merging complete!")


def merge_images_vertically(image1_path, image2_path, output_path, space_between=10):
    # Open the images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Calculate the size of the new image
    total_width = max(img1.width, img2.width)
    total_height = img1.height + img2.height + space_between

    # Create a new image with a white background
    merged_image = Image.new("RGB", (total_width, total_height), color="white")

    # Paste the first image
    merged_image.paste(img1, (0, 0))

    # Paste the second image below the first image with the specified space
    merged_image.paste(img2, (0, img1.height + space_between))

    # Save the merged image
    merged_image.save(output_path)

    print(f"Merged image saved to {output_path}")


def merge_images_from_directories(dir1, dir2, output_dir, space_between=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List and sort files naturally
    files1 = sorted(os.listdir(dir1), key=natural_sort_key)
    files2 = sorted(os.listdir(dir2), key=natural_sort_key)

    # Match the number of files in both directories
    num_files = min(len(files1), len(files2))
    files1 = files1[:num_files]
    files2 = files2[:num_files]

    for file1, file2 in zip(files1, files2):
        path1 = os.path.join(dir1, file1)
        path2 = os.path.join(dir2, file2)

        if os.path.isfile(path1) and os.path.isfile(path2):
            output_path = os.path.join(output_dir, f"merged_{file1}")

            merge_images_vertically(path1, path2, output_path, space_between)


def rename_images(directory):
    # Get all files in the directory
    files = os.listdir(directory)
    # Filter out only image files if needed, assuming all files in the directory are images
    images = [
        f
        for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
    ]
    # Sort the images using natural sort
    images.sort(key=natural_sort_key)

    # Rename images
    for idx, image in enumerate(images, start=1):
        # New filename with zero-padded index
        new_filename = f"image_{idx:05d}.png"
        # Full path for old and new file names
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_path, new_path)


def main():
    print("Hello World!")
    # merge_and_compare_images(
    #     "inputs/deform",
    #     "results_merged_images",
    #     "_v1024/restored_imgs",
    #     "v1024",
    #     "_v1.4/restored_imgs",
    #     "v1.4",
    #     "_vRF/restored_imgs",
    #     "_vRF",
    # )

    # merge_and_compare_images(
    #     "_v1024/cropped_faces",
    #     "results_merged_faces",
    #     "_v1024/restored_faces",
    #     "v1024",
    #     "_v1.4/restored_faces",
    #     "v1.4",
    #     "_vRF/restored_faces",
    #     "vRF",
    # )

    # rename_images("_v1024/cropped_faces_plus")
    # rename_images("_v1024/restored_faces_plus")
    # rename_images("_v1.4/restored_faces_plus")
    # rename_images("_vRF/restored_faces_plus")
    # merge_and_compare_images(
    #     "_v1024/cropped_faces_plus",
    #     "results_merged_faces_plus",
    #     "_v1024/restored_faces_plus",
    #     "v1024_plus",
    #     "_v1.4/restored_faces_plus",
    #     "v1.4_plus",
    #     "_vRF/restored_faces_plus",
    #     "vRF_plus",
    # )

    merge_images_from_directories(
        "/home/ubuntu/Desktop/eugene/GFPGAN/results_merged_faces",
        "/home/ubuntu/Desktop/eugene/GFPGAN/results_merged_faces_plus",
        "results_merged_final",
        space_between=20,
    )


if __name__ == "__main__":
    main()
