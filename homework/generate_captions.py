from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # 1. EGO CAR
    ego_kart = next((kart for kart in kart_objects if kart["is_center_kart"]), None)
    kart_name = ego_kart["kart_name"] if ego_kart else "An unknown kart"

    # 2. COUNTING
    num_karts = len(kart_objects)

    # 3. TRACK NAME
    track_name = extract_track_info(info_path)

    captions = [
        f"{kart_name} is the ego car.",
        f"There are {num_karts} karts in the scene.",
        f"The track is {track_name}."
    ]

    # 4. RELATIVE POSITION
    if ego_kart:
        for kart in kart_objects:
            if kart["is_center_kart"]:
                continue

            k_name = kart['kart_name']
            
            # Horizontal
            if kart["center"][0] < ego_kart["center"][0]:
                captions.append(f"{k_name} is left of the ego car.")
            else:
                captions.append(f"{k_name} is right of the ego car.")
                
            # Vertical
            if kart["center"][1] < ego_kart["center"][1]:
                captions.append(f"{k_name} is in front of the ego car.")
            else:
                captions.append(f"{k_name} is behind the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

def generate_dataset(data_dir: str = "data/train", output_name: str = "generated_captions.json"):
    """
    Generate the full QA dataset from all info files in a directory.
    """
    # assume this is run from Homework directory
    data_path = Path(data_dir)
    dataset = []

    info_files = list(data_path.glob("*_info.json"))
    for info_file in info_files:
        with open(info_file) as f:
            info = json.load(f)
        num_views = len(info["detections"])
        base_name = info_file.name.replace("_info.json", "")

        for view_index in range(num_views):
            img_name = f"{base_name}_{view_index:02d}_im.jpg"
            img_path = data_path / img_name
            if not img_path.exists():
                print(f"Warning: Image file {img_path} not found for info file {info_file}")
                continue
            
            try:
                captions = generate_caption(str(info_file), view_index)
                merged_caption = " ".join(captions)
                dataset.append({
                    "image_file": f"{data_path.name}/{img_name}",
                    "caption": merged_caption
                })
            except Exception as e:
                print(f"Error processing {info_file} view {view_index}: {e}")

    output_path = data_path / output_name
    # Save all QA pairs to a JSON file
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Successfully saved {len(dataset)} QA pairs to {output_path}")

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption,
               "generate_dataset": generate_dataset})


if __name__ == "__main__":
    main()
