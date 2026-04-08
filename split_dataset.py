import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Split BraTS 2024 dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed .npz files")
    parser.add_argument("--output", type=str, default="data/data_split.json", help="Output JSON path (default: data/data_split.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def get_timepoint_bin(count):
    return str(count) if count <= 5 else "6+"


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    files = sorted(data_dir.glob("*.npz"))
    assert len(files) > 0, f"No .npz files found in {data_dir}"
    print(f"Total files: {len(files)}")

    patient_to_files = defaultdict(list)
    for f in files:
        patient_id = f.stem.rsplit("-", 1)[0]
        patient_to_files[patient_id].append(f)

    patients = list(patient_to_files.keys())
    print(f"Total patients: {len(patients)}")

    timepoint_bins = [get_timepoint_bin(len(patient_to_files[p])) for p in patients]

    print(f"Timepoint distribution: {dict(Counter(timepoint_bins))}")

    patients_train, patients_temp, bins_train, bins_temp = train_test_split(
        patients, timepoint_bins,
        test_size=0.3, stratify=timepoint_bins, random_state=args.seed
    )
    patients_val, patients_test = train_test_split(
        patients_temp,
        test_size=0.5, stratify=bins_temp, random_state=args.seed
    )

    files_train = sorted([f for p in patients_train for f in patient_to_files[p]])
    files_val = sorted([f for p in patients_val for f in patient_to_files[p]])
    files_test = sorted([f for p in patients_test for f in patient_to_files[p]])

    assert set(patients_train) & set(patients_val) == set(), "Train/val overlap!"
    assert set(patients_train) & set(patients_test) == set(), "Train/test overlap!"
    assert set(patients_val) & set(patients_test) == set(), "Val/test overlap!"
    assert len(files_train) + len(files_val) + len(files_test) == len(files), "File count mismatch!"

    print(f"\n{'Split':<8} {'Patients':>10} {'Files':>10} {'%':>8}")
    print("-" * 40)
    for name, pts, fls in [
        ("Train", patients_train, files_train),
        ("Val", patients_val, files_val),
        ("Test", patients_test, files_test),
    ]:
        print(f"{name:<8} {len(pts):>10} {len(fls):>10} {len(pts)/len(patients)*100:>7.1f}%")
    print("-" * 40)
    print(f"{'Total':<8} {len(patients):>10} {len(files):>10} {'100.0%':>8}")

    split_data = {
        "train": [str(f) for f in files_train],
        "val": [str(f) for f in files_val],
        "test": [str(f) for f in files_test],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"\nSplit saved to: {output_path}")

if __name__ == "__main__":
    main()