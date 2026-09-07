"""Regenerate the synthetic ligand/pocket pair used by the smoke test.
Not real structures: they exercise the code paths, so predicted affinity is meaningless.
"""
import csv
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = Path(__file__).parent
RESIDUES = ["ALA", "ARG", "ASP", "PHE", "LEU", "TYR",
            "SER", "HIS", "VAL", "GLU", "TRP", "GLY"]


def main():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))   # paracetamol
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    with Chem.SDWriter(str(HERE / "ligand.sdf")) as w:
        w.write(mol)
    center = mol.GetConformer().GetPositions().mean(axis=0)

    rng = np.random.default_rng(0)
    lines, serial = [], 1
    for ri, rn in enumerate(RESIDUES):
        d = rng.normal(size=3)
        base = center + d / np.linalg.norm(d) * rng.uniform(5.0, 9.0)
        for name, el, off in [("N", "N", [0, 0, 0]), ("CA", "C", [1.45, 0, 0]),
                              ("C", "C", [2.0, 1.4, 0]), ("O", "O", [3.2, 1.5, 0]),
                              ("CB", "C", [1.9, -0.8, 1.2])]:
            p = base + np.array(off)
            lines.append(f"ATOM  {serial:5d}  {name:<3s} {rn} A{ri + 1:4d}    "
                         f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {el:>2s}")
            serial += 1
    (HERE / "pocket.pdb").write_text("\n".join(lines) + "\nEND\n")

    with (HERE / "pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "ligand_path", "pocket_path"])
        w.writerow(["example", "examples/ligand.sdf", "examples/pocket.pdb"])
    print(f"wrote ligand.sdf ({mol.GetNumAtoms()} atoms), "
          f"pocket.pdb ({len(RESIDUES)} residues), pairs.csv")


if __name__ == "__main__":
    main()
