import argparse
import logging
import math
import random
import re
import shutil
import typing as t
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw dataset")

    parser.add_argument(
        "--raw_dataset_dir",
        "-r",
        type=Path,
        required=True,
        help="Raw CelebA dataset directory",
    )
    parser.add_argument(
        "--output_dataset_dir",
        "-o",
        type=Path,
        help="Output dataset directory",
    )

    return parser.parse_args()


def create_dataset_structure(root_path: Path) -> None:
    logging.info(f"Creating folders structure in {root_path}...")

    train_path = root_path / "train" / "raw"
    val_path = root_path / "val" / "raw"
    test_path = root_path / "test" / "raw"

    for path in [train_path, val_path, test_path]:
        path.mkdir(parents=True, exist_ok=False)


def split_dataset(
    path: Path, out_path: Path, train_frac: float = 0.8, val_frac: float = 0.1
) -> None:
    logging.info("Splitting dataset...")

    abc_files = list((path / "ABC_cleaned").glob("*.abc"))

    for file in abc_files:
        train_list, val_list, test_list = split_elements(
            path, file, train_frac, val_frac
        )
        raw_file = open(path / "ABC_cleaned" / file.name, "r")
        out_file_train = open(out_path / "train" / "raw" / file.name, "w")
        out_file_val = open(out_path / "val" / "raw" / file.name, "w")
        out_file_test = open(out_path / "test" / "raw" / file.name, "w")

        lines = raw_file.readlines()
        index = 0
        max_index = len(lines)

        while index < max_index:
            match = re.search(r"X: (\d+)", lines[index])
            if match:
                if int(match.group(1)) in train_list:
                    out_file_train.write("\n")
                    while lines[index] != "\n":
                        out_file_train.write(lines[index])
                        index += 1
                    out_file_train.write(lines[index])
                    index += 1
                elif int(match.group(1)) in val_list:
                    out_file_val.write("\n")
                    while lines[index] != "\n":
                        out_file_val.write(lines[index])
                        index += 1
                    out_file_val.write(lines[index])
                    index += 1
                elif int(match.group(1)) in test_list:
                    out_file_test.write("\n")
                    while lines[index] != "\n":
                        out_file_test.write(lines[index])
                        index += 1
                    out_file_test.write(lines[index])
                    index += 1
            index += 1

        raw_file.close()
        out_file_train.close()
        out_file_val.close()
        out_file_test.close()


def split_elements(
    path: Path, file: str, train_frac: float = 0.8, val_frac: float = 0.1
) -> t.Tuple[list, list, list]:
    raw_file = open(path / "ABC_cleaned" / file.name, "r")

    lines = raw_file.readlines()
    num = 0

    for line in lines:
        match = re.search(r"X: (\d+)", line)
        if match:
            num = int(match.group(1))

    num_list = list(range(1, num + 1))
    random.shuffle(num_list)

    train_index = math.ceil(train_frac * num)
    val_index = math.ceil((train_frac + val_frac) * num)

    train_list = num_list[:train_index]
    val_list = num_list[train_index:val_index]
    test_list = num_list[val_index:]

    train_list = sorted(train_list)
    val_list = sorted(val_list)
    test_list = sorted(test_list)

    return train_list, val_list, test_list


def copy_files(path: Path, output_dset_root: Path) -> None:
    logging.info("Copying files...")

    abc_files = list((path / "ABC_cleaned").glob("*.abc"))

    for name_dataset in [
        "train",
        "val",
        "test",
    ]:
        for file in abc_files:
            shutil.copy(file, output_dset_root / name_dataset / "raw" / file.name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info("Building dataset...")
    args = parse_args()

    create_dataset_structure(args.output_dataset_dir)
    copy_files(args.raw_dataset_dir, args.output_dataset_dir)
    split_dataset(args.raw_dataset_dir, args.output_dataset_dir, 0.8, 0.1)

    logging.info("Finished building dataset.")


if __name__ == "__main__":
    main()
