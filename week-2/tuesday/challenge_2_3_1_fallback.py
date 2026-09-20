# challenge_2_3_1_fallback.py

from pathlib import Path


def run_resilient_ingestion():
    manual_data_dir = Path(__file__).resolve().parent / "manual_data"
    primary_path = manual_data_dir / "operational_guidelines_fake.txt"
    backup_path = manual_data_dir / "backup_guidelines.txt"

    # TODO: Step 1 - Try to load from primary_path
    try:
        with open(primary_path, "r") as file:
            data = file.read()
            print("Data loaded from primary path.")
    except FileNotFoundError:
        # TODO: Step 2 - Except FileNotFoundError, log a warning, and try loading
        # backup_path
        print(
            f"Warning: Primary file not found at {primary_path}. "
            "Attempting to load from backup."
        )
        try:
            with open(backup_path, "r") as backup_file:
                data = backup_file.read()
                print(f"Data loaded from backup path: {data}")
        except FileNotFoundError:
            # TODO: Step 3 - If backup fails too, raise a custom exception
            raise Exception(
                "Error: Both primary and backup files are missing. "
                f"Please check the paths: {primary_path}, {backup_path}."
            )


if __name__ == "__main__":
    run_resilient_ingestion()
