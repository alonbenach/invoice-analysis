# src/fc_map_utils.py
from pathlib import Path
import pandas as pd
import re
from unidecode import unidecode
from rapidfuzz import process, fuzz

# ------------------ Normalization ------------------

def normalize_text(s: pd.Series | list | pd.Index) -> pd.Series:
    """Lowercase, remove diacritics, keep [a-z0-9 + - space], squeeze spaces."""
    if not isinstance(s, pd.Series):
        s = pd.Series(list(s))
    out = (
        s.astype(str)
         .str.lower()
         .apply(unidecode)
         .str.replace(r"[^a-z0-9\s\-\+]", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )
    return out

def norm_one(x: str) -> str:
    return normalize_text(pd.Series([x])).iloc[0]

# ------------------ Category helpers ------------------

def infer_super_cat(text: str) -> str | None:
    """Map Polish menu text to broad FC super-category."""
    t = norm_one(text)
    if re.search(r"\bpizza\b", t): return "Pizza"
    if re.search(r"\bpinsa\b", t): return "Pizza"
    if re.search(r"\bhot\s?-?\s?dog\b|\bhotdog\b|\bhd\b", t): return "Hot Dog"
    if re.search(r"\bburger\b", t): return "Burger"
    if re.search(r"\bwrap\b", t): return "Wrap"
    if re.search(r"\bpanini\b|\bbagiet", t): return "Panini"
    if re.search(r"\btost(y|)\b", t): return "Tosty"
    if re.search(r"\bkanapk|\bsandw", t): return "Kanapki"
    if re.search(r"\bzapiekank", t): return "Zapiekanka"
    if re.search(r"\bsalatk|\bsalat|\bsalad", t): return "Sałatki"
    if re.search(r"\bfrytki\b|\bchrupbox\b|\bbox\b|\bnugget|\bstrip|\bfilecik|\bfilet\b", t): return "Frytki/Box"
    if re.search(r"\bkawa\b|\bespresso\b|\bamericano\b|\blatte\b|\bcappuccino\b|\bflat white\b", t): return "Kawa"
    return None

def hits_any(patterns, text) -> bool:
    return any(re.search(p, text) for p in patterns)

def is_weight_or_pack(text: str) -> bool:
    return bool(re.search(r"\b\d{2,4}\s?(g|kg|ml|l)\b", text) or re.search(r"\bx\s?\d+\b", text))

# ------------------ Rules (allow/deny) ------------------

# Positive stems per super-category (Polish stems included)
POS = {
    "Pizza":      [r"\bpizza\b", r"\bpinsa\b", r"\bpiec", r"\bpiecz", r"\bna wynos\b", r"\bslice\b", r"\bkawalek\b"],
    "Hot Dog":    [r"\bhot\s?-?\s?dog\b", r"\bhotdog\b", r"\bhd\b"],
    "Burger":     [r"\bburger\b"],
    "Wrap":       [r"\bwrap\b"],
    "Panini":     [r"\bpanini\b", r"\bbagiet"],
    "Tosty":      [r"\btost(y|)\b"],
    "Kanapki":    [r"\bkanapk", r"\bsandw(ich|)"],
    "Zapiekanka": [r"\bzapiekank"],
    "Sałatki":    [r"\bsalatk", r"\bsalat", r"\bsalad"],
    "Frytki/Box": [r"\bfrytki\b", r"\bchrupbox\b", r"\bbox\b", r"\bnugget", r"\bstrip", r"\bfilecik", r"\bfilet\b"],
    "Kawa":       [r"\bkawa\b", r"\bespresso\b", r"\bamericano\b", r"\blatte\b", r"\bcappuccino\b", r"\bflat white\b"],
}

# Obvious non-FC product_line cues (deny)
LINES_DENY = [r"\bnapoj\b", r"\bpiwo\b", r"\bpapieros", r"\blufka\b", r"\bbutelka\b"]

# product_line allow (mirror super-cats)
LINES_ALLOW = {
    "Pizza":      [r"\bpizza\b", r"\bpinsa\b"],
    "Hot Dog":    [r"\bhot\s?-?\s?dog\b", r"\bhotdog\b", r"\bhd\b"],
    "Burger":     [r"\bburger\b"],
    "Wrap":       [r"\bwrap\b"],
    "Panini":     [r"\bpanini\b", r"\bbagiet"],
    "Tosty":      [r"\btost(y|)\b"],
    "Kanapki":    [r"\bkanapk", r"\bsandw(ich|)"],
    "Zapiekanka": [r"\bzapiekank"],
    "Sałatki":    [r"\bsalatk", r"\bsalat", r"\bsalad"],
    "Frytki/Box": [r"\bfrytki\b", r"\bchrupbox\b", r"\bbox\b", r"\bnugget", r"\bstrip", r"\bfilecik", r"\bfilet\b"],
    "Kawa":       [r"\bkawa\b", r"\bespresso\b", r"\blatte\b", r"\bamericano\b", r"\bcappuccino\b"],
}

# Hard negatives: frozen/packaged & coffee-specific brand blockers
NEG_FROZEN_PACK = [r"\bmroz", r"\bmrozona", r"\bmrozone", r"\bzamroz", r"\bgleboko\b",
                   r"\bzgrzew", r"\bkarton\b", r"\bpaczka\b", r"\bkonserw", r"\bsloik\b"]
NEG_COFFEE = [r"\bnapoj( energetyczny|)\b", r"\benergy\b", r"\bmonster\b", r"\bred ?bull\b",
              r"\boshee\b", r"\b4move\b", r"\bpepsi\b", r"\bcoca\b", r"\bcola\b", r"\bfanta\b", r"\bsprite\b"]

def first_hit_category(text: str, cat2pats: dict) -> str | None:
    for cat, pats in cat2pats.items():
        if hits_any(pats, text):
            return cat
    return None

# ------------------ Main API ------------------

def map_fc_products(invoice_df: pd.DataFrame,
                    canonical_df: pd.DataFrame,
                    threshold: int = 70) -> pd.DataFrame:
    """
    Map UNIQUE invoice products to FC flag + best canonical item (within super-category).
    Returns columns:
      ['product_raw','product_norm','product_line','best_match_item','match_category','score','is_food_corner_auto']
    """
    # Canonical
    can = canonical_df.copy()
    if not {"menu_item","menu_category"}.issubset(can.columns):
        raise ValueError("Canonical must contain ['menu_item','menu_category']")
    can["menu_item_norm"] = normalize_text(can["menu_item"])
    can["menu_category_norm"] = normalize_text(can["menu_category"])
    # Broad bucket for subset matching
    can["super_cat"] = can.apply(lambda r: infer_super_cat(f"{r['menu_category']} {r['menu_item']}"), axis=1)

    # Unique invoice products (+ optional product_line)
    items = invoice_df[["product_name"]].dropna().drop_duplicates().copy()
    items["product_norm"] = normalize_text(items["product_name"])
    if "product_line" in invoice_df.columns:
        pl = (invoice_df[["product_name","product_line"]]
                .dropna()
                .assign(product_line=lambda d: normalize_text(d["product_line"]))
                .groupby("product_name")["product_line"]
                .agg(lambda s: s.value_counts().index[0])
                .reset_index())
        items = items.merge(pl, on="product_name", how="left")
    else:
        items["product_line"] = pd.NA

    out = []
    for _, row in items.iterrows():
        raw = row["product_name"]
        nm  = row["product_norm"]
        pln = str(row.get("product_line") or "")

        # 1) deny by frozen/packaged
        if hits_any(NEG_FROZEN_PACK, nm):
            out.append({**row, "best_match_item": None, "match_category": None, "score": 0, "is_food_corner_auto": False})
            continue

        # 2) obvious non-FC lines
        if pln and hits_any(LINES_DENY, pln):
            out.append({**row, "best_match_item": None, "match_category": None, "score": 0, "is_food_corner_auto": False})
            continue

        # 3) choose category (line hint → name POS → infer from text)
        cat = first_hit_category(pln, LINES_ALLOW) if pln else None
        if not cat:
            cat = first_hit_category(nm, POS)
        if not cat:
            cat = infer_super_cat(nm)

        # coffee-specific brand negatives
        if cat == "Kawa" and hits_any(NEG_COFFEE, nm):
            cat = None

        # weight/multipack: only exclude if no category was found
        if not cat and is_weight_or_pack(nm):
            out.append({**row, "best_match_item": None, "match_category": None, "score": 0, "is_food_corner_auto": False})
            continue

        if not cat:
            out.append({**row, "best_match_item": None, "match_category": None, "score": 0, "is_food_corner_auto": False})
            continue

        # 4) fuzzy inside the super-category derived from canonical
        subset = can[can["super_cat"] == cat]
        if subset.empty:
            out.append({**row, "best_match_item": None, "match_category": None, "score": 0, "is_food_corner_auto": False})
            continue

        subset_reset = subset.reset_index(drop=True)
        match = process.extractOne(nm, subset_reset["menu_item_norm"], scorer=fuzz.token_set_ratio)
        if not match:
            out.append({**row, "best_match_item": None, "match_category": cat, "score": 0, "is_food_corner_auto": False})
            continue

        score = int(match[1])
        best  = subset_reset.loc[match[2], "menu_item"]
        is_fc = score >= threshold

        out.append({**row, "best_match_item": best, "match_category": cat, "score": score, "is_food_corner_auto": is_fc})

    out_df = pd.DataFrame(out)[["product_name","product_norm","product_line","best_match_item","match_category","score","is_food_corner_auto"]]
    out_df.rename(columns={"product_name":"product_raw"}, inplace=True)
    return out_df
