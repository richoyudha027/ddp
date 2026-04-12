import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Sample subset from BraTS split")
    parser.add_argument(
        "--split_file", type=str, required=True,
        help="Path to existing data_split.json"
    )
    parser.add_argument(
        "--output", type=str, default="data/data_split_sample.json",
        help="Output JSON path (default: data/data_split_sample.json)"
    )
    parser.add_argument("--n_train", type=int, default=10, help="Number of train PATIENTS (default: 10)")
    parser.add_argument("--n_val", type=int, default=3, help="Number of val PATIENTS (default: 3)")
    parser.add_argument("--n_test", type=int, default=3, help="Number of test PATIENTS (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def group_by_patient(file_list):
    patient_to_files = defaultdict(list)
    for f in file_list:
        stem = Path(f).stem
        patient_id = stem.rsplit("-", 1)[0]
        patient_to_files[patient_id].append(f)
    return patient_to_files


def sample_patients(patient_to_files, n, rng):
    patients = list(patient_to_files.keys())
    n = min(n, len(patients))
    selected = rng.sample(patients, n)
    files = [f for p in selected for f in patient_to_files[p]]
    return selected, sorted(files)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.split_file, "r") as f:
        split = json.load(f)

    print(f"Original split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])} files")

    train_patients = group_by_patient(split["train"])
    val_patients = group_by_patient(split["val"])
    test_patients = group_by_patient(split["test"])

    print(f"Original patients: train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)}")

    sel_train, files_train = sample_patients(train_patients, args.n_train, rng)
    sel_val, files_val = sample_patients(val_patients, args.n_val, rng)
    sel_test, files_test = sample_patients(test_patients, args.n_test, rng)

    print(f"\nSampled subset:")
    print(f"  Train : {len(sel_train)} patients -> {len(files_train)} files")
    print(f"  Val   : {len(sel_val)} patients -> {len(files_val)} files")
    print(f"  Test  : {len(sel_test)} patients -> {len(files_test)} files")
    print(f"  Total : {len(files_train) + len(files_val) + len(files_test)} files")

    subset = {
        "train": files_train,
        "val": files_val,
        "test": files_test,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"\nSubset saved to: {output_path}")


if __name__ == "__main__":
    main()