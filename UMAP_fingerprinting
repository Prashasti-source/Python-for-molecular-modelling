Step 1: Upload your CSV file (run and upload phytochemicals_250.csv)
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
from rdkit.ML.Cluster import Butina
import umap
import matplotlib.pyplot as plt

# Step 2: Load your CSV (update filename if different)
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename, encoding='ISO-8859-1')

# Step 3: Define descriptor calculation function
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

descriptor_names = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 'RingCount']

# Step 4: Calculate descriptors and add to df
df[descriptor_names] = df['SMILES'].apply(lambda x: pd.Series(calculate_descriptors(x)))

# Step 5: Generate ECFP4 fingerprints
def compute_ecfp4(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(fp)
    return fps

smiles_list = df['SMILES'].dropna().tolist()
fps = compute_ecfp4(smiles_list)

# Filter out None fingerprints and track indices
valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
fps = [fp for fp in fps if fp is not None]

# Convert fingerprints to numpy array for clustering and visualization
def fps_to_numpy(fps_list):
    arr = np.zeros((len(fps_list), fps_list[0].GetNumBits()), dtype=np.int8)
    for i, fp in enumerate(fps_list):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr

fp_array = fps_to_numpy(fps)

# Step 6: Butina clustering on Tanimoto distance
def tanimoto_distance_matrix(fps):
    nfps = len(fps)
    dists = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    return dists

dists = tanimoto_distance_matrix(fps)
clusters = Butina.ClusterData(dists, len(fps), 0.3, isDistData=True)
print(f"Number of clusters found: {len(clusters)}")

# Map cluster labels back to original dataframe indices
cluster_labels = np.full(len(df), -1, dtype=int)
for cluster_id, cluster in enumerate(clusters):
    for idx in cluster:
        cluster_labels[valid_indices[idx]] = cluster_id
df['Cluster'] = cluster_labels

# Step 7: UMAP embedding for visualization
reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(fp_array)

plt.figure(figsize=(8,6))
scatter = plt.scatter(embedding[:,0], embedding[:,1], c=df.loc[valid_indices, 'Cluster'], cmap='tab20', s=50, alpha=0.8)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Chemical Space Visualization (UMAP)')
plt.grid(True)
plt.savefig('chemical_space_umap.png', dpi=300)
plt.show()

# Step 8: Save output CSV with descriptors and cluster info
output_csv = 'Chemical_Space_2.csv'
df.to_csv(output_csv, index=False)
print(f'Saved descriptor + cluster data to {output_csv}')
