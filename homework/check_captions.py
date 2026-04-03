import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check generated captions against grader MC QA dataset.")
    parser.add_argument("--generated", type=str, required=True, help="Path to your generated captions JSON")
    parser.add_argument("--grader", type=str, required=True, help="Path to the grader JSON (all_mc_qas.json)")
    args = parser.parse_args()

    with open(args.generated, 'r') as f:
        gen_data = json.load(f)
    with open(args.grader, 'r') as f:
        grader_data = json.load(f)

    # Convert generated data into a dictionary mapping filename to the list of captions
    # Using Path().name ensures we only compare the actual image filename (e.g., "00000_00_im.jpg")
    gen_dict = {}
    for item in gen_data:
        img_name = Path(item.get("image_file", item.get("image_path"))).name
        # Keep the captions as a list for subset checking
        gen_dict[img_name] = item.get("caption", [])

    matched = 0
    missing = []

    for grader_item in grader_data:
        img_name = Path(grader_item.get("image_file", "")).name
        
        # Grab the correct caption from the candidates list using the correct_index
        candidates = grader_item.get("candidates", [])
        answer_idx = grader_item.get("correct_index", 0)
        
        expected_caption = None
        if candidates and answer_idx < len(candidates):
            expected_caption = candidates[answer_idx]

        if not expected_caption:
            continue

        if img_name not in gen_dict:
            missing.append((img_name, expected_caption, "Image completely missing from generated data"))
            continue

        gen_captions = gen_dict[img_name]
        
        # Check if the exact good caption is one of the sentences you generated
        if expected_caption in gen_captions:
            matched += 1
        else:
            # Change good_caption to expected_caption here!
            missing.append((img_name, expected_caption, gen_captions))

    print(f"--- Caption Match Accuracy Check ---")
    print(f"Target Grader Images: {len(grader_data)}")
    print(f"Exact Matches Found:  {matched} ({(matched / len(grader_data)) * 100:.2f}%)")
    print(f"Mismatches:           {len(missing)}")

    if missing:
        print("\n--- Examples of Mismatches (Grader Expected VS Your Generated List) ---")
        for m in missing[:5]:
            print(f"\nImage: {m[0]}")
            print(f"  Expected:  {m[1]}")
            print(f"  Generated: {m[2]}")

if __name__ == "__main__":
    main()