import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check generated QA pairs against grader dataset.")
    parser.add_argument("--generated", type=str, required=True, help="Path to your generated JSON (e.g., data/valid/generated_qa_pairs.json)")
    parser.add_argument("--grader", type=str, required=True, help="Path to the grader JSON (e.g., data/valid_grader/balanced_qa_pairs.json)")
    args = parser.parse_args()

    with open(args.generated, 'r') as f:
        gen_data = json.load(f)
    with open(args.grader, 'r') as f:
        grader_data = json.load(f)

    # Convert lists of dicts to sets of tuples for set operations
    # Using .get for image path/file to handle potential key variations
    def to_set(data):
        return set(
            (item.get("image_file", item.get("image_path")), item["question"], item.get("answers", item.get("answer")))
            for item in data
        )

    gen_set = to_set(gen_data)
    grader_set = to_set(grader_data)

    matches = grader_set.intersection(gen_set)
    missing = grader_set - gen_set
    extra = gen_set - grader_set

    print(f"--- QA Data Accuracy Check ---")
    print(f"Target (Grader) QA Pairs: {len(grader_set)}")
    print(f"Your Generated QA Pairs:  {len(gen_set)}")
    
    match_pct = (len(matches) / len(grader_set)) * 100 if grader_set else 0
    print(f"Exact Matches:            {len(matches)} ({match_pct:.2f}%)")
    print(f"Missing (Under-generated):{len(missing)}")
    print(f"Extra (Over-generated):   {len(extra)}")

    if missing:
        print("\n--- Examples of MISSING QA pairs (In Grader, Not Yours) ---")
        for m in list(missing)[:3]:
            print(f"Image: {m[0]}\n Q: {m[1]}\n A: {m[2]}")

    if extra:
        print("\n--- Examples of EXTRA QA pairs (Yours, Not In Grader) ---")
        for e in list(extra)[:3]:
            print(f"Image: {e[0]}\n Q: {e[1]}\n A: {e[2]}")

if __name__ == "__main__":
    main()