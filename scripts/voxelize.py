# voxelize.py — voxelization with column name strip fix and extended features
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import pickle

# Parameters
grid_size = 20
resolution = 1.0
feature_channels = 10  # 8 atom types + 2 extra features

# Define atom types
ATOM_TYPES = ['C', 'O', 'N', 'S', 'P', 'H', 'F', 'Cl']
atom_to_channel = {atom: i for i, atom in enumerate(ATOM_TYPES)}

# Load matched Ki entries
csv_path = "data/filtered_ki_refined.csv"
data_root = "data/refined-set"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # Ensure no extra spaces in column headers

X = []
y = []

# Encode atom features
def atom_features(atom):
    vec = np.zeros(feature_channels)
    symbol = atom.GetSymbol() if hasattr(atom, 'GetSymbol') else atom.element
    if symbol in atom_to_channel:
        vec[atom_to_channel[symbol]] = 1
    # Extra feature 1: partial charge (ligand only, placeholder 0.0 for protein)
    if hasattr(atom, 'GetDoubleProp') and atom.HasProp('_GasteigerCharge'):
        try:
            vec[8] = float(atom.GetProp('_GasteigerCharge'))
        except:
            vec[8] = 0.0
    # Extra feature 2: atomic number
    try:
        vec[9] = atom.GetAtomicNum() if hasattr(atom, 'GetAtomicNum') else atom.element
    except:
        vec[9] = 0.0
    return vec

def place_atoms(coords, atoms):
    grid = np.zeros((grid_size, grid_size, grid_size, feature_channels))
    for coord, atom in zip(coords, atoms):
        i, j, k = (np.array(coord) / resolution).astype(int)
        if 0 <= i < grid_size and 0 <= j < grid_size and 0 <= k < grid_size:
            grid[i, j, k] = atom_features(atom)
    return grid

parser = PDBParser(QUIET=True)

for pdb_id, ki in zip(df["PDB_ID"], df["Ki"]):
    folder = os.path.join(data_root, pdb_id)
    prot_path = os.path.join(folder, f"{pdb_id}_protein.pdb")
    lig_path = os.path.join(folder, f"{pdb_id}_ligand.mol2")

    if not os.path.exists(prot_path) or not os.path.exists(lig_path):
        continue

    try:
        # --- Load Ligand ---
        mol = Chem.MolFromMol2File(lig_path, removeHs=False)
        if mol is None:
            continue
        AllChem.ComputeGasteigerCharges(mol)
        AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()
        ligand_coords = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        ligand_atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]

        # --- Load Protein ---
        structure = parser.get_structure(pdb_id, prot_path)
        protein_atoms = [atom for atom in structure.get_atoms() if atom.element != '']
        protein_coords = [atom.coord for atom in protein_atoms]

        # --- Voxelize ---
        lig_grid = place_atoms(ligand_coords, ligand_atoms)
        prot_grid = place_atoms(protein_coords, protein_atoms)
        grid = lig_grid + prot_grid

        X.append(grid)
        y.append(float(ki))

    except Exception as e:
        print(f"Skipping {pdb_id}: {e}")

# Save data
with open("data/train_grids.pkl", "wb") as f:
    pickle.dump(np.array(X), f)
with open("data/train_labels.pkl", "wb") as f:
    pickle.dump(np.array(y), f)

print(f"Saved {len(X)} voxel grids with extended features and labels.")
