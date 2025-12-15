BEGIN;

-- A) Raw invoices (September 2025)
-- Keep raw column names as-ingested for traceability.
CREATE TABLE IF NOT EXISTS raw_invoices_09_2025 (
    receipt_id               BIGINT,
    numer_paragonu           TEXT,
    data_zakupu              TEXT,
    godzina_zakupu           TEXT,
    linia_produktowa         TEXT,
    ean                      TEXT,
    nazwa_produktu           TEXT,
    qty                      NUMERIC,
    cena_jednostkowa_brutto  NUMERIC,
    stawka_vat               NUMERIC,
    cena_jednostkowa_netto   NUMERIC,
    rabat                    NUMERIC,
    kasjer                   TEXT,
    metoda_platnosci         TEXT,
    siec_sklepow             TEXT
);

-- B) Canonical Food Corner menu (hand-made reference)
CREATE TABLE IF NOT EXISTS canonical_menu (
    menu_category  TEXT NOT NULL,
    menu_item      TEXT NOT NULL,
    fc_type        TEXT NOT NULL
);

COMMIT;