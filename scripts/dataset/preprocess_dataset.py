import argparse
import logging
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw dataset")

    parser.add_argument(
        "--raw_dataset_dir",
        "-r",
        type=Path,
        required=True,
        help="Raw dataset directory",
    )

    return parser.parse_args()


def create_dataset(root_path: Path) -> None:
    logging.info(f"Creating folders structure in {root_path}...")

    train_path = root_path / "train" / "preprocessed"
    val_path = root_path / "val" / "preprocessed"
    test_path = root_path / "test" / "preprocessed"

    for path in [train_path, val_path, test_path]:
        path.mkdir(parents=True, exist_ok=False)


def preprocess_dataset(path: Path) -> None:

    train_path = path / "train"
    val_path = path / "val"
    test_path = path / "test"

    for path in [train_path, val_path, test_path]:
        for file in list(path.glob("*.abc")):
            raw_file = open(path / file.name, "r")
            preprocessed_file = open(path / "preprocessed" / file.name, "w")
            for line in raw_file:
                if len(line) > 1:
                    if (line[1] != ":" or line[0] == "|") and line[0] != "%":
                        preprocessed_file.write(line)
                else:
                    preprocessed_file.write(line)
            preprocessed_file = open(path / "preprocessed" / file.name, "w")
            raw_file.close()
            preprocessed_file.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()
    logging.info("Building preprocessed dataset...")
    create_dataset(args.raw_dataset_dir)
    preprocess_dataset(args.raw_dataset_dir)
    logging.info("Finished building preprocessed dataset.")


if __name__ == "__main__":
    main()
