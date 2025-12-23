from pathlib import Path
from src.extraction import process_all_locations

def main():
    BASE_DIR = Path(__file__).resolve().parents[1]

    process_all_locations(
        input_root=BASE_DIR / "data/raw/Train",
        output_dir=BASE_DIR / "data/process/extract",
        verbose=True
    )

if __name__ == "__main__":
    main()