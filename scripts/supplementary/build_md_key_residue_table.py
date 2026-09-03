"""Create the stringent multi-evidence MD key-residue table.

The input ``full_superset.csv`` contains the union of residues recovered by
five independently filtered MD evidence categories.  A residue is retained as
an MD key residue (also called a super-hub) only when at least two categories
support it.
"""

from pathlib import Path

import pandas as pd


INPUT = Path("full_superset.csv")
OUTPUT = Path("MD_key_residues_multi_evidence.csv")
CATEGORY_COLUMNS = ["Switch", "GCCM", "SB_hub", "Hydro_hub", "BC"]


def main() -> None:
    table = pd.read_csv(INPUT)
    missing = [column for column in CATEGORY_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"Missing category columns: {missing}")

    calculated_count = table[CATEGORY_COLUMNS].eq("Y").sum(axis=1)
    if "n" in table and not calculated_count.equals(table["n"]):
        raise ValueError("The stored category count does not match category flags")

    table = table.assign(n=calculated_count)
    key_residues = table.loc[table["n"] >= 2].copy()
    key_residues.to_csv(OUTPUT, index=False)

    category_sizes = {
        column: int(table[column].eq("Y").sum()) for column in CATEGORY_COLUMNS
    }
    print(f"Category sizes: {category_sizes}")
    print(f"Union: {len(table)} residues")
    print(f"Multi-evidence MD key residues (n >= 2): {len(key_residues)}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
