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
FC_CATEGORIES = {
    "Hot Dog", "Tosty", "Panini", "Frytki/Box", "Zupy",
    "Makaron", "Pizza", "Pinsa", "Sałatki"
}

_POS_PATTERNS = [
    r"\bhot[\s\-]?dog\b",
    r"\btost\b",
    r"\bpanini\b",
    r"\bfrytk",         # frytki / frytek
    r"\bnugget",        # nuggets
    r"\bkurczaker\b",   # brand-ish FC box
    r"\bstrips\b",
    r"\bfilecik",       # fileciki
    r"\bkurczaker\b",   # FC box name seen in data
]
_NEG_PATTERNS = [
    r"\bmro(ż|z)on\w*",           # frozen
    r"\bplastry\b",               # sliced cold cuts
    r"\bfilet\b.*\b\d{2,3}\s?g\b",
    r"\blisner\b|\bduda\b",       # packaged meat/salad brands
    r"\bpedigree\b|\breno\b",     # pet food
    r"\bduplo\b|\bsnickers\b",    # candy anchors
    r"\bpies\b|\bps(ów|ow)\b|\bkot\b"
]

_POS_RE = re.compile("|".join(_POS_PATTERNS), re.IGNORECASE)
_NEG_RE = re.compile("|".join(_NEG_PATTERNS), re.IGNORECASE)

def _rule_is_fc(product_line_norm: str) -> bool:
    if not isinstance(product_line_norm, str):
        return False
    return bool(_POS_RE.search(product_line_norm)) and not bool(_NEG_RE.search(product_line_norm))

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

def map_fc_products(
    invoice_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    threshold: int = 70,
    key_col: str = "product_line",
) -> pd.DataFrame:
    """
    Deterministically + fuzzily map UNIQUE invoice products to Food Corner (FC).
    - Primary ID is `key_col` (default: product_line).
    - FC decision: rule-first on product_line, then category-aware fuzzy match.
    Returns columns:
      ['product_raw','product_norm','product_line','best_match_item',
       'match_category','score','is_food_corner_auto']
    """

    # --------- configuration (self-contained) ----------
    import re
    from rapidfuzz import fuzz, process

    FC_CATEGORIES = {
        "Hot Dog", "Tosty", "Panini", "Frytki/Box", "Zupy",
        "Makaron", "Pizza", "Pinsa", "Sałatki"
    }

    POS_PATTERNS = [
        r"\bhot[\s\-]?dog\b",
        r"\btost(y)?\b|\btoast\b",
        r"\bpanini\b",
        r"\bfrytk\w*\b",
        r"\bnugget\w*\b",
        r"\bstrips?\b",
        r"\bfilecik\w*\b",
        r"\bkurczaker\b",
    ]
    NEG_PATTERNS = [
        r"\bmro(ż|z)on\w*",                 # frozen
        r"\bplastry\b",                    # sliced cold cuts
        r"\bfilet\b.*\b\d{2,3}\s?g\b",     # fixed-weight packaged meats
        r"\blisner\b|\bduda\b",            # packaged meat/salad brands
        r"\bpedigree\b|\breno\b",          # pet food
        r"\bduplo\b|\bsnickers\b",         # candy anchors
        r"\bpies\b|\bps(ów|ow)\b|\bkot\b", # animal terms
    ]

    POS_RE = re.compile("|".join(POS_PATTERNS), re.IGNORECASE)
    NEG_RE = re.compile("|".join(NEG_PATTERNS), re.IGNORECASE)

    def rule_is_fc(product_line_norm: str) -> bool:
        if not isinstance(product_line_norm, str):
            return False
        return bool(POS_RE.search(product_line_norm)) and not bool(NEG_RE.search(product_line_norm))

    # --------- canonical prep ----------
    can = canonical_df.copy()
    req_cols = {"menu_item", "menu_category"}
    if not req_cols.issubset(can.columns):
        raise ValueError(f"Canonical must contain {sorted(req_cols)}")
    can["menu_item_norm"] = normalize_text(can["menu_item"])
    # use a tiny list to speed up rapidfuzz calls
    can_items = can["menu_item_norm"].tolist()

    # --------- unique invoice items ----------
    if key_col not in invoice_df.columns:
        raise ValueError(f"key_col '{key_col}' not found in invoice_df columns")

    items = (
        invoice_df[[key_col]]
        .dropna()
        .drop_duplicates()
        .rename(columns={key_col: "product_key_raw"})
    )
    items["product_norm"] = normalize_text(items["product_key_raw"])

    # normalized product_line column for rule logic
    if key_col == "product_line":
        items["product_line"] = normalize_text(items["product_key_raw"])
    else:
        # fallback: try to take a representative normalized product_line if present
        if "product_line" in invoice_df.columns:
            pl = (
                invoice_df[[key_col, "product_line"]]
                .dropna()
                .assign(product_line=lambda d: d["product_line"].apply(normalize_text))
                .groupby(key_col, as_index=False)["product_line"]
                .agg(lambda s: s.value_counts().index[0])
                .rename(columns={key_col: "product_key_raw"})
            )
            items = items.merge(pl, on="product_key_raw", how="left")
        else:
            items["product_line"] = pd.NA

    # --------- per-item mapping ----------
    out = []
    for _, r in items.iterrows():
        raw = r["product_key_raw"]
        nm  = r["product_norm"]
        pln = r.get("product_line")
        pln = pln if isinstance(pln, str) else ""

        # deterministic rule on product_line
        rule_fc = rule_is_fc(pln)

        # fuzzy to all canonical items
        best_item = None
        match_cat = None
        score = 0
        if len(can_items) > 0:
            match = process.extractOne(nm, can_items, scorer=fuzz.token_set_ratio)
            if match:
                score = int(match[1]) if match[1] is not None else 0
                idx = int(match[2]) if match[2] is not None else -1
                if 0 <= idx < len(can):
                    best_item = can.iloc[idx]["menu_item"]
                    match_cat = can.iloc[idx]["menu_category"]

        # category-aware / threshold decision
        cat_fc = (match_cat in FC_CATEGORIES) and (score >= 60)
        is_fc  = bool(rule_fc or cat_fc or (score >= threshold))

        out.append({
            "product_key_raw": raw,
            "product_norm": nm,
            "product_line": pln,
            "best_match_item": best_item,
            "match_category": match_cat,
            "score": score,
            "is_food_corner_auto": is_fc,
        })

    out_df = pd.DataFrame(out)[
        ["product_key_raw","product_norm","product_line",
         "best_match_item","match_category","score","is_food_corner_auto"]
    ]
    out_df.rename(columns={"product_key_raw": "product_raw"}, inplace=True)
    return out_df

