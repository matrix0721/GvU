import os
import json
import re

def numeric_sort_key(filename):
    # 提取文件名前面的数字部分
    match = re.match(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return float("inf")  # 没有数字的放最后

def reorganize(image_folder, metadata_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # 读取所有metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_lines = f.readlines()

    # 假设images/ 下已经是 N × M 张图片（每条metadata对应M张图片）
    all_images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg"))],
        key=numeric_sort_key
    )

    num_metadata = len(metadata_lines)
    num_images = len(all_images)
    # 自动推断每个metadata对应多少张图
    imgs_per_metadata = num_images // num_metadata

    print(f"Found {num_metadata} metadata entries, {num_images} images.")
    print(f"Each entry gets {imgs_per_metadata} images.")

    idx = 0
    for i, line in enumerate(metadata_lines):
        sample_dir = os.path.join(output_folder, f"{i:05d}")
        samples_subdir = os.path.join(sample_dir, "samples")
        os.makedirs(samples_subdir, exist_ok=True)

        # 写metadata.jsonl（一行）
        with open(os.path.join(sample_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
            f.write(line.strip() + "\n")

        # 拷贝对应的图片
        for j in range(imgs_per_metadata):
            img_name = all_images[idx]
            idx += 1
            src = os.path.join(image_folder, img_name)
            dst = os.path.join(samples_subdir, f"{j:04d}.png")
            os.system(f"cp '{src}' '{dst}'")

        # 如果有 grid.png，可以额外处理
        grid_src = os.path.join(image_folder, f"grid_{i:05d}.png")
        if os.path.exists(grid_src):
            os.system(f"cp '{grid_src}' '{os.path.join(sample_dir, 'grid.png')}'")

    print("Done!")

if __name__ == "__main__":
    reorganize(
        image_folder="Image Primary Folder", 
        metadata_file="/geneval/prompts/evaluation_metadata.jsonl", 
        output_folder="Image Output Folder"
    )
