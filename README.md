# iDualG4: A dual-channel deep learning framework for predicting in vivo G-quadruplexes

## 📖 Introduction

G-quadruplexes (G4s) are non-canonical nucleic acid secondary structures that help maintain genomic stability and regulate gene transcription. Although the genome contains a vast number of putative G4-forming sequences (PQSs), only a small fraction fold stably into G4 structures within the complex chromatin environment of living cells. 

Existing deep-learning approaches improve predictive accuracy by incorporating cell line-specific epigenetic data; however, their heavy reliance on costly, large-scale sequencing assays (e.g., ChIP-seq) limits broader application to clinical samples and newly profiled cell lines. 

To address this challenge, we propose **iDualG4**, an interpretable dual-channel deep learning framework that relies solely on DNA sequences, thereby obviating the need for additional cell-specific sequencing data. Leveraging the Enformer pre-trained module, iDualG4 infers epigenomic contexts directly from DNA sequences and integrates this information with a local sequence feature channel, facilitating the precise prediction of in vivo G4 structures. 

Evaluations across multiple cell lines, including K562, demonstrate that iDualG4 significantly outperforms existing methods, particularly in handling imbalanced data (achieving an AUPR of 0.981 on K562). Interpretability analysis based on DeepSHAP indicates that iDualG4 provides an in vivo G4 prediction tool combining high precision and interpretability without the need for additional experimental sequencing data, and offers a novel computational framework for elucidating how sequence and the epigenetic environment jointly determine genomic G4 formation.

---

## ⚙️ Required Packages

* python 3.8
* tensorflow-gpu 2.5.0
* keras 2.5.0
* numpy 1.19.5
* scikit-learn 1.2.1
* scipy 1.9.3


---

## 🚀 Quick Start

### 1. Data Preparation
Our dual-channel model requires two types of inputs. Please ensure the following data files are prepared:
* **DNA Sequences (`.fa`):** FASTA files for sequence one-hot encoding (e.g., `pos_grich.fa`, `neg_plus.fa`).
* **Epigenetic Features (`.npy`):** NumPy arrays extracted using the Enformer pre-trained module (e.g., `pos_enformer.npy`).

*Note: Please update the file paths in `model.py` to match your local data directory before running.*

### 2. Training and Evaluation
The `model.py` script includes the complete pipeline for data processing, model building, and 5-fold cross-validation. Run the following command to start training:

```bash
python model.py
