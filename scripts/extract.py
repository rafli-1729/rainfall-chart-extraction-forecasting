import logging
from pathlib import Path
from src.extraction import process_all_locations

from src.config import RAW_DIR, PROCESS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    process_all_locations(
        input_root=RAW_DIR/"Train",
        output_dir=PROCESS_DIR/"extract",
        verbose=True
    )

if __name__ == "__main__":
    main()