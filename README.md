# 🧬 Protein-Ligand Binding Affinity Prediction Using 3D Convolutional Neural Networks

This project aims to predict the **binding affinity (Ki)** between proteins and ligands using raw structural data (`.mol2` and `.pdb`) and a **3D Convolutional Neural Network (CNN)** trained on voxelized features. It eliminates traditional docking steps and allows an end-to-end deep learning-based regression approach.

---

## 🔍 Overview

- **Input:** `.mol2` ligand file, `.pdb` protein file, and matched experimental `Ki` values.
- **Voxelization:** Structural files are converted into 3D voxel grids capturing atomic and chemical features.
- **Model:** A 3D CNN processes the voxel grid to regress the binding affinity.
- **Output:** Predicted Ki values, evaluation metrics (RMSE, R²), and visualizations (scatter plot, histogram).

---

## 📁 Project Structure

```
protein-ligand-binding-affinity-using-cnn/
├── data/                      # Dataset & Ki CSVs
│   ├── refined-set/          # Contains .mol2 and .pdb files per PDB ID
│   ├── train_grids.pkl       # Voxelized 3D grid data
│   └── train_labels.pkl      # Corresponding Ki values
│
├── models/                   # Trained model (.keras or .h5)
│
├── outputs/                  # Generated visualizations and CSVs
│   ├── scatter_true_vs_pred.png
│   ├── hist_prediction_error.png
│   └── predictions_vs_actual.csv
│
├── scripts/
│   ├── voxelize.py           # Converts .pdb + .mol2 to voxel grid
│   ├── train_cnn.py          # CNN training and model saving
│   └── visualize_training.py # Prediction + plotting + export CSV
│
├── .gitignore
└── README.md
```

---

## 🧪 Features Used for Voxelization

Each atom (from protein or ligand) is voxelized into a 20×20×20 3D grid with **10 feature channels**, including:

- One-hot encoding of atom type (C, O, N, S, P, H, F, Cl)
- Gasteiger partial charge
- Atomic number

---

## 🧠 Model Architecture (3D CNN)

- `Conv3D` layers to extract spatial patterns in atomic grids  
- `BatchNormalization` + `ReLU` for stability and non-linearity  
- `MaxPooling3D` for downsampling  
- `Dropout` layers to reduce overfitting  
- `Dense` layers for final regression output (Ki value)

---

## 🏃‍♀️ How to Run

### 1. Voxelize the Dataset

```bash
python scripts/voxelize.py
```

### 2. Train the Model

```bash
python scripts/train_cnn.py
```

### 3. Evaluate & Visualize

```bash
python scripts/visualize_training.py
```

---

## 📊 Sample Evaluation Output

| Metric | Value |
|--------|-------|
| RMSE   | 1.54  |
 

---

## 🧠 No Docking or Pocket Detection

Unlike traditional pipelines, this method:
- Uses raw 3D structure (.pdb + .mol2) directly
- Does not extract binding pockets manually
- Learns binding regions implicitly via CNN

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries:
- TensorFlow / Keras
- RDKit
- Biopython
- NumPy / Pandas / Matplotlib
- Scikit-learn

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 👩‍🔬 Credits
  
Dataset: [PDBBind v2020 Refined Set](http://www.pdbbind.org.cn/)

---

## 📌 Acknowledgments

Thanks to:
- Open Babel & RDKit for molecule processing  
- TensorFlow for deep learning tools  
- PDBBind for curated Ki data
