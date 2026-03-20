# Install RDKit and other dependencies in Colab
# 1. Upload file (if not already uploaded)
from google.colab import files
uploaded = files.upload()

# 2. Install RDKit if not done already
!pip install -q condacolab
import condacolab
condacolab.install()  # Runtime will restart, re-run from top after this step

# After restart, install RDKit
!mamba install -c conda-forge rdkit -y

# 3. Import libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 4. Load your CSV file
df = pd.read_csv('phytochemicals_250.csv', encoding_errors='ignore')  # Make sure file is in current directory

# 5. Calculate descriptors function
def calculate_descriptors(smiles):
    if not isinstance(smiles, str):
        return [None]*7
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None]*7
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)
    return [mw, logp, tpsa, hbd, hba, rotb, rings]

df[descriptor_names] = df['SMILES'].apply(lambda x: pd.Series(calculate_descriptors(x)))

# Assuming SMILES column is named 'SMILES'
df[descriptor_names] = df['SMILES'].apply(lambda x: pd.Series(calculate_descriptors(x)))

# 6. Save new CSV (Supplementary Table S2)
df.to_csv('Chemical_Space.csv', index=False)
print("Chemical_Space file saved!")

# 7. PCA and scatter plot chemical space (Figure S1)
desc_df = df[descriptor_names].dropna()

pca = PCA(n_components=2)
components = pca.fit_transform(desc_df)

plt.figure(figsize=(8,6))
plt.scatter(components[:,0], components[:,1], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Chemical Space Scatter Plot')
plt.grid(True)
plt.savefig('chemical_space.png', dpi=300)
plt.show()

# 8. Optional: Download outputs
files.download('Chemical_Space.csv')
files.download('chemical_space.png')
