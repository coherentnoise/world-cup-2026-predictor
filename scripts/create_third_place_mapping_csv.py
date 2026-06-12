# create_third_place_mapping_csv.py
# Build data/third_place_mapping.csv from the online third-place mapping table.

from pathlib import Path
import pandas as pd

SOURCE_URL = "https://en.wikipedia.org/wiki/Template:2026_FIFA_World_Cup_third-place_table"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "third_place_mapping.csv"


def clean_cell(value):
    value = str(value).strip().replace(" ", "")
    return value.upper()


def main():
    tables = pd.read_html(SOURCE_URL)

    best = None
    for table in tables:
        flat_cols = [str(c) for c in table.columns]
        text = " ".join(flat_cols).lower()
        if "1a" in text or "qualified" in text or "slot" in text:
            if len(table) >= 100:
                best = table
                break

    if best is None:
        raise RuntimeError("Could not identify the third-place mapping table.")

    # This page/table can change formatting, so we normalize fairly defensively.
    df = best.copy()
    df.columns = [str(c).split("'")[-2] if "'" in str(c) else str(c) for c in df.columns]
    df.columns = [c.strip() for c in df.columns]

    # If the source table is already close to our schema, keep those columns.
    rename = {}
    for c in df.columns:
        lc = c.lower().replace(" ", "")
        if "qualified" in lc or "combination" in lc or lc in {"groups", "thirdplace"}:
            rename[c] = "qualified_groups"
        elif lc in {"1a", "slot_1a"}:
            rename[c] = "slot_1A"
        elif lc in {"1b", "slot_1b"}:
            rename[c] = "slot_1B"
        elif lc in {"1d", "slot_1d"}:
            rename[c] = "slot_1D"
        elif lc in {"1e", "slot_1e"}:
            rename[c] = "slot_1E"
        elif lc in {"1g", "slot_1g"}:
            rename[c] = "slot_1G"
        elif lc in {"1i", "slot_1i"}:
            rename[c] = "slot_1I"
        elif lc in {"1k", "slot_1k"}:
            rename[c] = "slot_1K"
        elif lc in {"1l", "slot_1l"}:
            rename[c] = "slot_1L"

    df = df.rename(columns=rename)

    required = ["qualified_groups", "slot_1A", "slot_1B", "slot_1D", "slot_1E", "slot_1G", "slot_1I", "slot_1K", "slot_1L"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Could not parse required columns: {missing}. Inspect the source table format.")

    out = df[required].copy()
    for c in required:
        out[c] = out[c].map(clean_cell)
    out["qualified_groups"] = out["qualified_groups"].map(lambda x: "".join(sorted(x.replace(",", ""))))
    out.insert(0, "option", range(1, len(out) + 1))
    out = out.drop_duplicates("qualified_groups")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUTPUT_PATH}")
    if len(out) != 495:
        print("WARNING: expected 495 rows. Check the source table formatting.")


if __name__ == "__main__":
    main()
