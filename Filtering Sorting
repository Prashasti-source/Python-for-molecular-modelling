pip install rdkit-pypi pandas tqdm numpy

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from tqdm import tqdm
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    input_file: str = "library.sdf"
    output_file: str = "filtered_compounds.csv"
    max_violations: int = 1   # allowed Lipinski violations

# load molecules
def load_molecules(file):
    logger.info(f"Loading molecules from {file}")

    if file.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(file)
        mols = [mol for mol in suppl if mol is not None]

    elif file.endswith(".csv"):
        df = pd.read_csv(file)
        mols = [Chem.MolFromSmiles(sm) for sm in df['SMILES']]

    else:
        raise ValueError("Unsupported file format")

    logger.info(f"Loaded {len(mols)} molecules")
    return mols


# descriptor 

def compute_descriptors(mol):
    try:
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "MR": Crippen.MolMR(mol)
        }
    except:
        return None


# filtering rules

def lipinski_rule(desc):
    violations = 0
    if desc["MW"] > 500: violations += 1
    if desc["LogP"] > 5: violations += 1
    if desc["HBD"] > 5: violations += 1
    if desc["HBA"] > 10: violations += 1
    return violations

def veber_rule(desc):
    return desc["TPSA"] <= 140 and desc["RotB"] <= 10

def ghose_rule(desc):
    return (
        160 <= desc["MW"] <= 480 and
        -0.4 <= desc["LogP"] <= 5.6 and
        20 <= desc["HeavyAtoms"] <= 70 and
        40 <= desc["MR"] <= 130
    )

def toxicity_filter(desc):
    # Simple heuristic toxicity flags
    if desc["LogP"] > 6:
        return False
    if desc["TPSA"] < 20:
        return False
    return True

def druglikeness_score(desc):
    score = 0
    if desc["MW"] < 500: score += 1
    if desc["LogP"] < 5: score += 1
    if desc["TPSA"] < 140: score += 1
    if desc["RotB"] < 10: score += 1
    return score

def ligand_efficiency(score, heavy_atoms):
    if heavy_atoms == 0:
        return None
    return score / heavy_atoms

def filter_library(mols, config):

    results = []

    for i, mol in enumerate(tqdm(mols)):

        desc = compute_descriptors(mol)
        if desc is None:
            continue

        lipinski_viol = lipinski_rule(desc)
        veber = veber_rule(desc)
        ghose = ghose_rule(desc)
        tox = toxicity_filter(desc)

        drug_score = druglikeness_score(desc)

        # Placeholder docking score (replace later)
        docking_score = -7.0

        le = ligand_efficiency(docking_score, desc["HeavyAtoms"])

        passed = (
            lipinski_viol <= config.max_violations and
            veber and
            ghose and
            tox
        )

        results.append({
            "ID": i,
            **desc,
            "Lipinski_Violations": lipinski_viol,
            "Veber": veber,
            "Ghose": ghose,
            "Toxicity": tox,
            "DrugScore": drug_score,
            "LigandEfficiency": le,
            "Pass": passed
        })

    df = pd.DataFrame(results)

    # Filter passing compounds
    filtered_df = df[df["Pass"] == True]

    logger.info(f"Filtered {len(filtered_df)} compounds out of {len(df)}")

    filtered_df.to_csv(config.output_file, index=False)

    return df, filtered_df


# exceution

if __name__ == "__main__":

    config = FilterConfig(
        input_file="zinc_library.sdf",
        output_file="filtered_hits.csv",
        max_violations=1
    )

    molecules = load_molecules(config.input_file)

    full_df, filtered_df = filter_library(molecules, config)

    print("\nTop filtered compounds:")
    print(filtered_df.head())
