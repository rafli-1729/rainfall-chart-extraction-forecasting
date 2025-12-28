import logging
from pathlib import Path
from src.plot_extraction import process_all_locations

from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    process_all_locations(
        input_root=config.paths.raw/"Train",
        output_dir=config.paths.interim/"extracted",
        verbose=True
    )

if __name__ == "__main__":
    main()