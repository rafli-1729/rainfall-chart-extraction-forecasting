import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    run("python -m scripts.data.build_api_dataset")
    run("python -m scripts.data.init_database")