from pathlib import Path
import pandas as pd

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def list_csvs(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.glob("*.csv")])

def read_csv(path: str | Path, encoding: str | None = "utf-8") -> pd.DataFrame:
    # Relaxed read; polish locales often mix separators/encodings across shards.
    # If you see decoding errors, try encoding="cp1250".
    return pd.read_csv(path, encoding=encoding)

def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
