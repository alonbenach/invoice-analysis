import pandas as pd
from datetime import datetime, time
from pathlib import Path
import yaml
import re

# ---------- Column normalization ----------

ALIASES = {
    # receipts
    "id paragonu": "receipt_id",
    "id_paragonu": "receipt_id",
    "paragon_id": "receipt_id",
    "numer_paragonu": "receipt_number",

    # date/time
    "data zakupu": "purchase_date",
    "data_zakupu": "purchase_date",
    "data": "purchase_date",
    "godzina zakupu": "purchase_time",
    "godzina_zakupu": "purchase_time",
    "godzina": "purchase_time",

    # product & price
    "ean": "ean",
    "nazwa produktu": "product_name",
    "nazwa_produktu": "product_name",
    "nazwa": "product_name",
    "linia_produktowa": "product_line",

    "ilość": "qty",
    "ilosc": "qty",
    "ilosc_sztuk": "qty",
    "ilość sztuk": "qty",

    "cena jednostkowa brutto": "unit_price_gross",
    "cena_jednostkowa_brutto": "unit_price_gross",
    "cena brutto": "unit_price_gross",
    "wartosc brutto": "unit_price_gross",
    "wartość brutto": "unit_price_gross",

    "cena_jednostkowa_netto": "unit_price_net",
    "stawka_vat": "vat_rate",
    "rabat": "discount",

    # cashier/payment
    "kasjer": "cashier",
    "cashier": "cashier",
    "metoda płatności": "payment_method",
    "metoda platnosci": "payment_method",
    "metoda_platnosci": "payment_method",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and map Polish headers to normalized English ones."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns}, inplace=True)
    return df

# ---------- Clean non-product entries ----------
NON_PRODUCTS = ["torba", "butelka", "opak", "kompania", "reklam", "kubek"]

def is_non_product(name: str) -> bool:
    if not isinstance(name, str):
        return False
    lower = name.lower()
    return any(pat in lower for pat in NON_PRODUCTS)

# ---------- Numeric casting ----------

def _to_number_series(s: pd.Series) -> pd.Series:
    """Robust number parser for PL formats (1 234,56 → 1234.56)."""
    if s.dtype.kind in "biufc":
        return s
    s = s.astype(str).str.replace(r"\s", "", regex=True)  # remove spaces
    s = s.str.replace(",", ".", regex=False)              # comma → dot
    return pd.to_numeric(s, errors="coerce")

def cast_basic_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "qty" in df.columns:
        df["qty"] = _to_number_series(df["qty"])
    if "unit_price_gross" in df.columns:
        df["unit_price_gross"] = _to_number_series(df["unit_price_gross"])
    if "unit_price_net" in df.columns:
        df["unit_price_net"] = _to_number_series(df["unit_price_net"])
    if "vat_rate" in df.columns:
        df["vat_rate"] = _to_number_series(df["vat_rate"])
    if "discount" in df.columns:
        df["discount"] = _to_number_series(df["discount"])
    return df

def normalize_ean(df: pd.DataFrame) -> pd.DataFrame:
    """Make EAN a clean string while preserving NA."""
    df = df.copy()
    if "ean" in df.columns:
        # Convert floats like 5901234567890.0 → "5901234567890"
        try:
            df["ean"] = df["ean"].astype("Int64").astype(str)
        except Exception:
            df["ean"] = df["ean"].astype(str)
        df["ean"] = df["ean"].str.replace("<NA>", "", regex=False).str.strip()
    return df

# ---------- Timestamp & slots ----------

def _normalize_time_token(x: str) -> str:
    """Accept messy time tokens: '8:5', '815', '0815', '123045'."""
    if ":" in x:
        return x
    if not x.isdigit():
        return x
    n = len(x)
    if n <= 2:   # "5" -> "00:05"
        return f"00:{x.zfill(2)}"
    if n == 3:   # "815" -> "08:15"
        return f"0{x[0]}:{x[1:]}"
    if n == 4:   # "0815" -> "08:15"
        return f"{x[:2]}:{x[2:]}"
    if n == 6:   # "123045" -> "12:30:45"
        return f"{x[:2]}:{x[2:4]}:{x[4:]}"
    return x

def _normalize_time_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(pd.NaT, index=[])
    s = s.astype(str).str.strip().map(_normalize_time_token)
    return pd.to_datetime(s, errors="coerce").dt.time

def parse_timestamp(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    Build df['ts']:
    - Combine normalized purchase_date + purchase_time when available.
    - Fallback to purchase_date only.
    - Localize to timezone; supports older pandas without errors= kw.
    Also creates df['hour_minute'] ("%H:%M").
    """
    df = df.copy()

    # Date column (must exist after normalize_columns)
    date_ser = pd.to_datetime(df.get("purchase_date"), errors="coerce", dayfirst=True)

    # Optional time column
    if "purchase_time" in df.columns:
        time_ser = _normalize_time_series(df["purchase_time"])
        if len(time_ser) != len(df):
            time_ser = pd.Series(pd.NaT, index=df.index)
    else:
        time_ser = pd.Series(pd.NaT, index=df.index)

    # Combine if any time exists, else rely on date only
    if time_ser.notna().any():
        combo = pd.to_datetime(
            date_ser.dt.strftime("%Y-%m-%d") + " " + pd.Series(time_ser).astype(str),
            errors="coerce"
        )
    else:
        combo = date_ser

    ts = combo

    # Localize to tz (pandas-version safe)
    try:
        ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    except TypeError:
        try:
            ts = ts.dt.tz_localize(tz)
        except Exception:
            pass
    except Exception:
        pass

    df["ts"] = ts
    df["hour_minute"] = df["ts"].dt.strftime("%H:%M")
    return df

def _to_time(hhmm: str) -> time:
    return datetime.strptime(hhmm, "%H:%M").time()

def assign_slots(df: pd.DataFrame, slots_cfg_path: str | Path) -> pd.DataFrame:
    """Map df['ts'] to slot_id/slot_label using YAML config ranges."""
    df = df.copy()
    cfg = yaml.safe_load(Path(slots_cfg_path).read_text(encoding="utf-8"))
    slots = cfg["slots"]

    mm = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    slot_ids, slot_labels = [], []

    for m in mm.fillna(-1):
        sid, lab = None, None
        if m == -1:
            slot_ids.append(None); slot_labels.append(None); continue
        for s in slots:
            start = _to_time(s["start"]); end = _to_time(s["end"])
            sm, em = start.hour*60 + start.minute, end.hour*60 + end.minute
            if sm <= m < em:
                sid, lab = s["id"], s["label"]; break
        slot_ids.append(sid); slot_labels.append(lab)

    df["slot_id"] = slot_ids
    df["slot_label"] = slot_labels
    return df

# ---------- Basic audit summary ----------

def basic_checks(df: pd.DataFrame) -> dict:
    out = {"rows": len(df), "cols": len(df.columns)}
    for c in ["receipt_id","purchase_date","purchase_time","ean","product_name","qty","unit_price_gross"]:
        if c in df.columns:
            out[f"nulls_{c}"] = int(df[c].isna().sum())
    return out

def clean_non_products(series , non_products) -> pd.Series:
    clean_index = [
        idx for idx in series.index
        if not any(word in idx.lower() for word in NON_PRODUCTS)
    ]
    return series.loc[clean_index]