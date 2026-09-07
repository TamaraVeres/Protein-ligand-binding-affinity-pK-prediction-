"""Check that each ligand/pocket file parses, then attach its path to the table.
"""
import csv
from pathlib import Path
from Bio.PDB import PDBParser
from rdkit import Chem

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PL_ROOT = DATA_DIR / "P-L"
IN_CSV = DATA_DIR / "pl_table.csv"
OUT_CSV = DATA_DIR / "pl_table_with_paths.csv"


def pocket_is_valid(pocket_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pocket", str(pocket_path))
    except Exception:
        return False
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() == "HOH":
                    continue
                if res.has_id("CA"):
                    return True
    return False


def ligand_is_valid(ligand_path):
    path = str(ligand_path)
    try:
        if path.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
            mol = suppl[0]
        elif path.endswith(".mol2"):
            mol = Chem.MolFromMol2File(path, removeHs=False, sanitize=False)
        else:
            return False
        if mol is None or mol.GetNumAtoms() == 0:
            return False
        if mol.GetNumConformers() == 0:
            return False
        return True
    except Exception:
        return False


def build_path_index():
    pdb_to_paths = {}

    for ligand in PL_ROOT.rglob("*_ligand.sdf"):
        pdb_id = ligand.stem.split("_")[0]
        entry = pdb_to_paths.setdefault(pdb_id, {"ligand": None, "pocket": None})
        entry["ligand"] = ligand

    for ligand in PL_ROOT.rglob("*_ligand.mol2"):
        pdb_id = ligand.stem.split("_")[0]
        entry = pdb_to_paths.setdefault(pdb_id, {"ligand": None, "pocket": None})
        if entry["ligand"] is None:
            entry["ligand"] = ligand

    for pocket in PL_ROOT.rglob("*_pocket.pdb"):
        pdb_id = pocket.stem.split("_")[0]
        entry = pdb_to_paths.setdefault(pdb_id, {"ligand": None, "pocket": None})
        entry["pocket"] = pocket

    return pdb_to_paths


def main():
    pdb_to_paths = build_path_index()
    print(f"Indexed {len(pdb_to_paths)} PDB IDs from P-L")

    rows = []
    skipped = 0
    with IN_CSV.open() as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        for row in reader:
            pdb_id = row["pdb_id"]
            paths = pdb_to_paths.get(pdb_id, {"ligand": None, "pocket": None})

            ligand = paths["ligand"]
            pocket = paths["pocket"]

            if ligand is None or pocket is None:
                row["ligand_path"] = ""
                row["pocket_path"] = ""
                rows.append(row)
                continue

            if not pocket_is_valid(pocket) or not ligand_is_valid(ligand):
                skipped += 1
                row["ligand_path"] = ""
                row["pocket_path"] = ""
                rows.append(row)
                continue

            row["ligand_path"] = str(ligand.relative_to(SCRIPT_DIR))
            row["pocket_path"] = str(pocket.relative_to(SCRIPT_DIR))
            rows.append(row)

    with OUT_CSV.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(1 for r in rows if r["ligand_path"] and r["pocket_path"])
    print(f"Wrote {len(rows)} rows ({valid} valid, {skipped} skipped bad structures) to {OUT_CSV}")


if __name__ == "__main__":
    main()