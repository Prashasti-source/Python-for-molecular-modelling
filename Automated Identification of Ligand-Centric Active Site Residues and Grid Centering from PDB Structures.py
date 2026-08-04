# Get the grid size and active site residues around the ligand via Colab
!pip install biopython

from Bio.PDB import PDBParser, NeighborSearch
import numpy as np

# Upload your PDB file first
from google.colab import files
uploaded = files.upload()
pdb_file = list(uploaded.keys())[0]

parser = PDBParser()
structure = parser.get_structure("6B1E", pdb_file) # Write PDB ID for your desired protein

ligand_resname = "LF7"  # Use actual ligand resname here

# Get ligand atoms
ligand_atoms = []
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.get_resname() == ligand_resname:
                for atom in residue:
                    ligand_atoms.append(atom)

if not ligand_atoms:
   raise ValueError(f"Ligand {ligand_resname} not found")

# Calculate ligand centroid (grid center)
coords = np.array([atom.get_coord() for atom in ligand_atoms])
center = coords.mean(axis=0)
print("Ligand centroid (grid center):", center)

# Get all atoms excluding ligand and excluding waters (HOH)
all_atoms = [atom for atom in structure.get_atoms() if atom.get_parent().get_resname() != ligand_resname and atom.get_parent().get_resname() != "HOH"]

ns = NeighborSearch(all_atoms)

cutoff = 5.0
close_residues = set()

for latom in ligand_atoms:
    for neighbor in ns.search(latom.get_coord(), cutoff):
        res = neighbor.get_parent()
        chain_id = res.get_parent().id
        res_id = res.id[1]
        res_name = res.get_resname()
        close_residues.add((chain_id, res_name, res_id))

print("Active site residues within 5 Å of ligand excluding waters:")
for chain_id, res_name, res_id in sorted(close_residues):
    print(f"Chain {chain_id}, Residue {res_name} {res_id}")
