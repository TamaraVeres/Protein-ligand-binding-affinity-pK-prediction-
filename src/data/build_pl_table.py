"""Build data/pl_table.csv from the raw PDBbind index, applying the curation filters.
"""
import csv
import math
from pathlib import Path

# Repo-root/data, where the committed tables live.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
INDEX_PATH = DATA_DIR / "index" / "INDEX_general_PL.2020R1.lst"
PL_ROOT = DATA_DIR / "P-L"
OUT_CSV = DATA_DIR / "pl_table.csv"


def convert_to_molar(value_str, unit_str):
    v = float(value_str)
    u = unit_str.lower()

    if u == "m":
        return v
    if u == "mm":
        return v * 1e-3
    if u == "um":
        return v * 1e-6
    if u == "nm":
        return v * 1e-9
    if u == "pm":
        return v * 1e-12
    return None


def find_ligand_and_pocket_paths(pdb_id):
    ligand_path = None
    pocket_path = None

    for folder in PL_ROOT.rglob(pdb_id):
        if not folder.is_dir():
            continue

        sdf = list(folder.glob(f"{pdb_id}_ligand.sdf"))
        mol2 = list(folder.glob(f"{pdb_id}_ligand.mol2"))
        if sdf:
            ligand_path = sdf[0]
        elif mol2:
            ligand_path = mol2[0]

        pockets = list(folder.glob(f"{pdb_id}_pocket.pdb"))
        if pockets:
            pocket_path = pockets[0]

        if ligand_path or pocket_path:
            break

    return ligand_path, pocket_path


def add_affinity(affinity):
    if affinity.startswith("Ki"):
        assay_type = "Ki"
        rest = affinity[2:]
    elif affinity.startswith("Kd"):
        assay_type = "Kd"
        rest = affinity[2:]
    elif affinity.startswith("IC50"):
        assay_type = "IC50"
        rest = affinity[4:]
    else:
        return None

    if not rest:
        return None

    comparison_sign = rest[0] 
    value_and_unit = rest[1:]

    value = []
    unit = []
    for character in value_and_unit:
        if character.isdigit() or character == '.':
            value.append(character)
        else:
            unit.append(character)

    if not value or not unit:
        return None

    value_str = "".join(value)
    unit_str = "".join(unit)

    return assay_type, comparison_sign, value_str, unit_str


def main():
    rows = []

    with INDEX_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            pdb_id = parts[0]
            resolution = parts[1]
            year = parts[2]
            affinity = parts[3]

            comment = ""
            if "//" in line:
                comment = line.split("//", 1)[1].strip()
            
            ligand_id = ""
            if "(" in comment and ")" in comment:
                start = comment.rfind("(") + 1
                end = comment.find(")", start)
                if end != -1:
                    ligand_id = comment[start:end].strip()

            assay_type, comparison_sign, value_str, unit_str = add_affinity(affinity)
            if assay_type not in ["Ki", "Kd"]:
                continue

            if comparison_sign != "=":
                continue

            c = comment.lower()
            if "incomplete" in c and "ligand" in c:
                continue
            if "covalent" in c:
                continue

            if resolution == "NMR":
                continue
            try:
                res_val = float(resolution)
            except ValueError:
                continue
            if res_val > 2.5:
                continue

            value_M = convert_to_molar(value_str, unit_str)
            p_value = None
            if value_M is not None:
                p_value = -math.log10(value_M)

            if p_value is None or p_value < 2.0 or p_value > 12.0:
                continue

            ligand_path = ""
            pocket_path = ""

            rows.append({
                "pdb_id": pdb_id,
                "resolution": resolution,
                "year": year,
                "assay_type": assay_type,
                "comparison_sign": comparison_sign,
                "value_raw": value_str,
                "unit_raw": unit_str,
                "value_M": value_M,
                "p_value": p_value,
                "ligand_path": str(ligand_path) if ligand_path else "",
                "pocket_path": str(pocket_path) if pocket_path else "",
                "comment": comment,
                "ligand_id": ligand_id,
            })

    fieldnames = [
        "pdb_id", "resolution", "year",
        "assay_type", "comparison_sign", "value_raw", "unit_raw",
        "value_M", "p_value", "ligand_id",
        "ligand_path", "pocket_path",
        "comment",
    ]

    with OUT_CSV.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()