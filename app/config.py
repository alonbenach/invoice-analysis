from pathlib import Path
import os

# Project root (adjust if your structure is different)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw input files
INVOICES_CSV = PROJECT_ROOT / "data" / "invoices" / "invoices_09_2025.csv"
CANONICAL_CSV = PROJECT_ROOT / "data" / "refs" / "zabka_food_corner_menu_canonical.csv"

# Neon connection string comes from environment variable NEON_DSN
NEON_DSN = os.environ.get("NEON_DSN")

if NEON_DSN is None:
    raise RuntimeError(
        "Environment variable NEON_DSN is not set. "
        "Export it in your shell with your Neon connection string."
    )