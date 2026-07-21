# QETHOS — Quantum ETHereum On-chain Security

QETHOS explores whether quantum machine learning (QML) offers any practical edge over
classical models for detecting fraudulent Ethereum/DeFi transactions. The project trains
and evaluates two quantum classifiers — a **Variational Quantum Classifier (VQC)** and a
**Quantum Kernel SVM (QSVM)** — against classical baselines (Random Forest, XGBoost, SVM)
on the same balanced dataset, and stress-tests both under adversarial attacks.

Author: Frank Junior Hoza Longfor

## Repository layout

```
Dataset/
  ctgan_balanced_400k_dataset.csv     400k-row balanced transaction dataset (9 features + label)
  code/
    QETHOS_CTGAN_Training.ipynb       Trains a CTGAN to synthesize minority-class fraud
                                      samples and builds the balanced dataset above
QSVM/
  QETHOS_QSVM_Angle.ipynb             Quantum-kernel SVM using an angle-embedding fidelity kernel
VQC/
  QETHOS_VQC_Pipeline.ipynb           Variational quantum classifier with trainable
                                      data re-uploading encoding
LICENSE                               MIT License
```

## Dataset

`Dataset/ctgan_balanced_400k_dataset.csv` contains 400,000 rows (200,000 fraud / 200,000
legitimate) derived from the **BCCC-DeFiFraudTrans-2025** transaction data, balanced with
CTGAN-generated synthetic fraud samples. Each row has 9 "AGA" (address/gas/amount) features
plus a binary label:

| Column | Description |
|---|---|
| `length_to` | Length of the recipient address field |
| `block_number` | Ethereum block number of the transaction |
| `gas_used` | Gas consumed by the transaction |
| `chain_id` | Chain identifier |
| `total_gas_cost` | Total gas cost (gas × price) |
| `effective_gas_price` | Effective gas price paid |
| `cumulative_gas_used` | Cumulative gas used in the block up to this transaction |
| `gas_price_ratio` | Ratio of paid gas price to a baseline/reference price |
| `length_log` | Log-scaled address/field length |
| `flag` | Label — `1` = fraudulent, `0` = legitimate |

### `Dataset/code/QETHOS_CTGAN_Training.ipynb`
Trains a Conditional Tabular GAN (CTGAN) on the real (imbalanced) fraud transactions and
generates ~111k synthetic fraud samples to balance the classes before QML training. Includes
statistical quality checks comparing real vs. synthetic feature distributions and saves both
the synthetic data and the trained CTGAN model.

## Models

### `QSVM/QETHOS_QSVM_Angle.ipynb` — Quantum Kernel SVM
Builds a fidelity-based quantum kernel via angle embedding (PennyLane) and trains an SVM
(`SVC(kernel="precomputed")`) on the resulting Gram matrix. Highlights:
- Preprocessing: `StandardScaler → optional PCA → QuantileTransformer` into the angle range
- A hyperparameter sweep (angle scale × rotation axis × layers) scored by Kernel Target
  Alignment (KTA) to find a well-conditioned (non-degenerate) kernel before the full build
- Kernel diagnostics (off-diagonal concentration, KTA) to catch a collapsed or trivial kernel
- Parallelized, pickle-safe kernel construction for HPC/cluster use
- Multi-seed stability runs, publication-ready circuit figures, and classical baseline comparison
- Adversarial robustness evaluation (ART) against gradient-free attacks

### `VQC/QETHOS_VQC_Pipeline.ipynb` — Variational Quantum Classifier
A trainable PyTorch + PennyLane (`TorchLayer`) hybrid model using data re-uploading with
trainable per-layer encoding scale/shift and dual-axis rotations. Highlights:
- Quantile (rank) encoding of features instead of RobustScaler+tanh, to avoid angle
  saturation on extreme (fraud-indicative) gas values
- 70/15/15 train/validation/test split; decision threshold tuned on validation only,
  reported once on a held-out test set
- Validation-F1-based checkpointing to select the best-generalizing epoch
- Classical baselines (Random Forest, XGBoost) trained on identical data for a fair
  head-to-head comparison, plus an optional "full 400k" classical reference run
- Adversarial robustness evaluation (ART), mirroring the QSVM notebook's methodology

## Running the notebooks

Both quantum notebooks auto-detect their environment (Google Colab vs. a persistent
JupyterHub/NRP volume) and mount storage accordingly. To run:

1. Open a notebook in Google Colab (or a compatible Jupyter environment).
2. Run the environment/storage-setup cell first — it mounts Google Drive on Colab or
   points to a local persistent path otherwise.
3. Run the dependency-install cell (PennyLane, PennyLane-Lightning, scikit-learn, XGBoost,
   PyTorch, `adversarial-robustness-toolbox`, etc.).
4. Point the configured paths at `Dataset/ctgan_balanced_400k_dataset.csv` (or regenerate
   it from `Dataset/code/QETHOS_CTGAN_Training.ipynb` first).
5. Run cells top to bottom. The QSVM notebook's hyperparameter sweep (Section 12) is
   meant to be run before the full kernel build to pick a working configuration cheaply.

## License

MIT — see [LICENSE](LICENSE).
