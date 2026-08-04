# Install required packages if running in Colab
!pip install pandas requests tqdm

import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import quote
from google.colab import files

# Prompt user to upload a CSV file
print("Please upload your .csv file containing compound names...")
uploaded = files.upload()

# Automatically takes the first uploaded file
csv_filename = next(iter(uploaded.keys()))
df = pd.read_csv(csv_filename)

# Automatically try to use the first column, or change this to your column name
compound_names = df[df.columns[0]].dropna().astype(str).unique().tolist()

def get_pubchem_info(name):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/CanonicalSMILES/JSON'
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            props = data['PropertyTable']['Properties'][0]
            cid = props.get('CID', '')
            smiles = props.get('CanonicalSMILES', '')
            return cid, smiles
    except Exception as e:
        return '', ''
    return '', ''

results = []
for name in tqdm(compound_names):
    cid, smiles = get_pubchem_info(name)
    results.append({'Compound Name': name, 'PubChem CID': cid, 'SMILES': smiles})

result_df = pd.DataFrame(results)
result_df.to_csv('phytochemicals_with_pubchem.csv', index=False)
print(result_df.head(15))

# Optional: download the result file
files.download('phytochemicals_with_pubchem.csv')
