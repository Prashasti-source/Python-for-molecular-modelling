pip install vina biopython rdkit-pypi meeko matplotlib

import os
import subprocess
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List

from vina import Vina
import py3Dmol
from rdkit import Chem
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class DockConfig:
    receptor: str = "protein.pdb"
    ligand: str = "ligand.sdf"
    out_dir: str = "dock_out"

    center = (10, 10, 10)
    size = (20, 20, 20)

    exhaustiveness = 16
    poses = 10

# docking

class Docking:

    def __init__(self, cfg: DockConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

    def prepare(self):
        logger.info("Preparing receptor and ligand...")

        subprocess.run(f"prepare_receptor4.py -r {self.cfg.receptor} -o {self.cfg.out_dir}/rec.pdbqt", shell=True)
        subprocess.run(f"prepare_ligand4.py -l {self.cfg.ligand} -o {self.cfg.out_dir}/lig.pdbqt", shell=True)

    def run(self):
        v = Vina(sf_name='vina')

        v.set_receptor(f"{self.cfg.out_dir}/rec.pdbqt")
        v.set_ligand_from_file(f"{self.cfg.out_dir}/lig.pdbqt")

        v.compute_vina_maps(center=self.cfg.center, box_size=self.cfg.size)

        v.dock(exhaustiveness=self.cfg.exhaustiveness, n_poses=self.cfg.poses)

        out_file = f"{self.cfg.out_dir}/docked.pdbqt"
        v.write_poses(out_file, n_poses=self.cfg.poses)

        logger.info("Docking complete.")
        return out_file

# score extraction

def extract_affinity(pdbqt):
    scores = []
    with open(pdbqt) as f:
        for line in f:
            if "REMARK VINA RESULT" in line:
                scores.append(float(line.split()[3]))
    return scores

# plip analysis

def run_plip(complex_pdb):
    logger.info("Running PLIP analysis...")
    subprocess.run(f"plip -f {complex_pdb} -o plip_out", shell=True)

def parse_plip():
    import xml.etree.ElementTree as ET

    interactions = {
        "H-bonds": 0,
        "Hydrophobic": 0,
        "Pi-stacking": 0,
        "Salt bridges": 0
    }

    xml_file = "plip_out/report.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for hb in root.iter('hydrogen_bond'):
        interactions["H-bonds"] += 1

    for hp in root.iter('hydrophobic_interaction'):
        interactions["Hydrophobic"] += 1

    for pi in root.iter('pi_stack'):
        interactions["Pi-stacking"] += 1

    for sb in root.iter('salt_bridge'):
        interactions["Salt bridges"] += 1

    return interactions

# 3D visualization

def show_3d(pdb_file):
    with open(pdb_file) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')

    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.addStyle({'resn': 'LIG'}, {'stick': {'colorscheme': 'greenCarbon'}})

    view.zoomTo()
    return view

# 2D plots

def plot_interactions(interactions: Dict):
    labels = list(interactions.keys())
    values = list(interactions.values())

    plt.figure()
    plt.bar(labels, values)
    plt.title("Protein-Ligand Interactions")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("interaction_plot.png")
    plt.show()

class FullDockingWorkflow:

    def __init__(self, cfg: DockConfig):
        self.cfg = cfg

    def run(self):

        dock = Docking(self.cfg)

        dock.prepare()
        docked_file = dock.run()

        # Extract scores
        scores = extract_affinity(docked_file)

        print("\nBinding Affinities (kcal/mol):", scores)
        print("Best Affinity:", min(scores))

        # Convert best pose to PDB
        subprocess.run(f"obabel {docked_file} -O complex.pdb", shell=True)

        # PLIP analysis
        run_plip("complex.pdb")
        interactions = parse_plip()

        print("\nInteraction Summary:")
        for k, v in interactions.items():
            print(f"{k}: {v}")

        # 2D plot
        plot_interactions(interactions)

        # 3D visualization
        view = show_3d("complex.pdb")
        view.show()

# execute

if __name__ == "__main__":

    cfg = DockConfig(
        receptor="protein.pdb",
        ligand="ligand.sdf",
        center=(10,10,10),
        size=(20,20,20)
    )

    workflow = FullDockingWorkflow(cfg)
    workflow.run()
