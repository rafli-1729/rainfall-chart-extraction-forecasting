from warnings import filterwarnings
filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import config
from src.dataset_builder import 

def load_single_month(path):
    data = pd.read_csv(path)
    return data

def main():


if __name__ == "__main__":
    ...
