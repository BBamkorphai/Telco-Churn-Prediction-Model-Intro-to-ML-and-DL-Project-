from pathlib import Path
import os

def target_path(join_path: str):
    BASE_DIR = Path.cwd()
    return os.path.join(BASE_DIR, join_path)

def save_csv(df, save_path: str):
    df.to_csv(target_path(save_path), index=False)