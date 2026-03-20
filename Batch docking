pip install vina rdkit-pypi pandas tqdm

import os
import subprocess
import logging
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

from rdkit import Chem
from vina import Vina

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    receptor_pdb: str = "protein.pdb"
    ligand_sdf: str = "library.sdf"
    out_dir: str = "batch_results"

    center = (10, 10, 10)
    size = (20, 20, 20)

    exhaustiveness = 16
    num_modes = 5

# prepare receptor

def prepare_receptor(pdb, out):
    if not os.path.exists(out):
        logger.info("Preparing receptor...")
        subprocess.run(f"prepare_receptor4.py -r {pdb} -o {out}", shell=True, check=True)

# sdf splitting

def split_sdf(sdf_file: str) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(sdf_file)
    mols = [mol for mol in suppl if mol is not None]
    logger.info(f"Loaded {len(mols)} ligands from SDF.")
    return mols

# prepare ligand

def prepare_ligand(mol, index, out_dir):
    pdb_file = os.path.join(out_dir, f"lig_{index}.pdb")
    pdbqt_file = os.path.join(out_dir, f"lig_{index}.pdbqt")

    Chem.MolToPDBFile(mol, pdb_file)

    subprocess.run(f"prepare_ligand4.py -l {pdb_file} -o {pdbqt_file}", shell=True, check=True)

    return pdbqt_file

# dock single ligand

def dock_ligand(vina_obj, ligand_pdbqt, out_file):
    vina_obj.set_ligand_from_file(ligand_pdbqt)

    vina_obj.dock(exhaustiveness=16, n_poses=5)
    vina_obj.write_poses(out_file, n_poses=1)

    return extract_score(out_file)

# extract score

def extract_score(pdbqt):
    with open(pdbqt) as f:
        for line in f:
            if "REMARK VINA RESULT" in line:
                return float(line.split()[3])
    return None

# batching

class BatchDocking:

    def __init__(self, cfg: BatchConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        self.receptor_pdbqt = os.path.join(cfg.out_dir, "receptor.pdbqt")

        prepare_receptor(cfg.receptor_pdb, self.receptor_pdbqt)

        self.vina = Vina(sf_name='vina')
        self.vina.set_receptor(self.receptor_pdbqt)

        self.vina.compute_vina_maps(
            center=cfg.center,
            box_size=cfg.size
        )

    def run(self):

        mols = split_sdf(self.cfg.ligand_sdf)

        results = []

        for i, mol in enumerate(tqdm(mols)):

            try:
                ligand_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lig_{i}"

                ligand_pdbqt = prepare_ligand(mol, i, self.cfg.out_dir)

                out_file = os.path.join(self.cfg.out_dir, f"dock_{i}.pdbqt")

                score = dock_ligand(self.vina, ligand_pdbqt, out_file)

                results.append({
                    "Ligand_ID": ligand_id,
                    "Score (kcal/mol)": score,
                    "Pose_File": out_file
                })

            except Exception as e:
                logger.warning(f"Failed ligand {i}: {e}")
                continue

        df = pd.DataFrame(results)

        # Rank results
        df = df.sort_values(by="Score (kcal/mol)")

        csv_path = os.path.join(self.cfg.out_dir, "docking_results.csv")
        df.to_csv(csv_path, index=False)

        logger.info(f"Docking complete. Results saved to {csv_path}")

        print("\nTop 10 hits:")
        print(df.head(10))


# execuion

if __name__ == "__main__":

    cfg = BatchConfig(
        receptor_pdb="protein.pdb",
        ligand_sdf="zinc_library.sdf",
        center=(10,10,10),
        size=(20,20,20),
        exhaustiveness=32
    )

    workflow = BatchDocking(cfg)
    workflow.run()
