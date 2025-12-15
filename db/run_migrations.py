from pathlib import Path
import os
import psycopg2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = PROJECT_ROOT / "db" / "migrations"

DSN = os.environ.get("NEON_DSN")
if not DSN:
    raise RuntimeError("NEON_DSN is not set in the environment.")

def main():
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"No migration files found in {MIGRATIONS_DIR}")

    print(f"Found {len(files)} migrations.")
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    for f in files:
        sql = f.read_text(encoding="utf-8")
        print(f"Running migration: {f.name}")
        cur.execute(sql)
        conn.commit()

    cur.close()
    conn.close()
    print("All migrations applied successfully.")

if __name__ == "__main__":
    main()