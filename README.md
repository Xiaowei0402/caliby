<img src="assets/caliby.jpg" alt="Caliby" width="175"/>

# Caliby
This is a forked repo of Caliby for faster in-memory implementation.

## Caliby Wrapper Installation (Recommended)
To use the `Caliby` wrapper class from anywhere, you have two options:

### Option A: Direct Installation (Simplest)
Install directly from GitHub without cloning:
```bash
pip install "git+https://github.com/Xiaowei0402/caliby.git@main"
```

### Option B: Local Development Installation
Clone the repo if you need to run examples or modify the code:
```bash
git clone https://github.com/Xiaowei0402/caliby.git
cd caliby
pip install -e .
```

### Handling PyArrow Conflicts
Some versions of `atomworks` strictly pin `pyarrow==17.0.0`, which can conflict with other requirements in your environment (e.g. `pyarrow>=21.0.0`). 

If you encounter installation errors related to `pyarrow` and `atomworks`:
1. Install `caliby` dependencies while ignoring `atomworks`'s rigid pins:
   ```bash
   pip install . --no-deps
   # Then manually install other dependencies or:
   pip install torch lightning hydra-core omegaconf biotite numpy pandas scipy tqdm rdkit zstandard einops PyYAML torchtyping p_tqdm
   ```
2. For the best compatibility with modern Arrow versions, you can install `atomworks` via:
   ```bash
   pip install atomworks==2.2.0 --no-deps
   ```

### Post-Installation Setup
The `caliby` package automatically handles the environment variables required by `atomworks` (`PDB_MIRROR_PATH` and `CCD_MIRROR_PATH`) upon import. 

However, if you are running scripts in this repository directly without importing the `caliby` package, or if you encounter issues, you should manually set them in your terminal or activation script:

```bash
# Manual setup (copy-paste in terminal or add to ~/.bashrc)
export PDB_MIRROR_PATH=""
export CCD_MIRROR_PATH=""

# Or add to your venv activation script for automatic loading
echo 'export PDB_MIRROR_PATH=""' >> path/to/your/venv/bin/activate
echo 'export CCD_MIRROR_PATH=""' >> path/to/your/venv/bin/activate
```

## Quick Usage

You can now use the `Caliby` wrapper class as a pip-installable package. Model weights will be **automatically downloaded** to `~/.cache/caliby/` the first time you initialize the model (approx. 400MB).

```python
import pandas as pd
from caliby import Caliby

# 1. Initialize the model
# Auto-detects weights or downloads them automatically if missing.
model = Caliby(device="cuda", seed=0)

# Optionally load the soluble model instead
# model = Caliby(use_soluble=True, device="cuda")

# Load your structure content
with open("examples/example_data/native_pdbs/7xz3.cif", "r") as f:
    pdb_content = f.read()

# 1. Basic Design
results = model.design(pdb_content, num_seqs=2, temperature=0.1)
print(f"Designed sequence: {results[0]['seq']}")

# 2. Design with Constraints
# resid: index in label_seq_id
constraints = pd.DataFrame([{
    "fixed_pos_seq": "A1-10", 
    "fixed_pos_override_seq": "A20:C",
    "pos_restrict_aatype": "A21:FL"
}])
results_constrained = model.design(pdb_content, pos_constraint_df=constraints)

# 3. Ensemble Design
# First string in the list is treated as the primary conformer
ensemble = [pdb_content, open("conformer_1.pdb").read()]
results_ensemble = model.design_ensemble(ensemble, num_seqs=2)

# 4. Scoring
score_info = model.score(pdb_content)
print(f"Energy (U): {score_info['scores']['U']}")
```

### Position-wise Constraints (pos_constraint_df)

When using the `Caliby` wrapper, position-wise constraints can be passed as a `pandas.DataFrame`. 

| Column Name | Example Format | Description |
|-------------|----------------|-------------|
| `fixed_pos_seq` | `A1-10,B1-5` | Residues to keep fixed (identity conditioned on input). |
| `fixed_pos_scn` | `A1,A3` | Sidechains to keep fixed (must be subset of `fixed_pos_seq`). |
| `fixed_pos_override_seq` | `A5:G,A6:L` | Override input residue to specific AA and fix it. |
| `pos_restrict_aatype` | `A10:AVG` | Restrict allowed AA types at specific positions. |
| `symmetry_pos` | `A1,B1\|A2,B2` | Tie sampling weights between positions (e.g. for homooligomers). |

**Note on Indexing**: Residue indices should follow the `label_seq_id` (Cif/mmCif standard). In PyMOL, you can view these by running `set cif_use_auth, off` before loading.

### In-memory Ensembles

The `design_ensemble` and `score_ensemble` methods accept a **list of strings** (`list[str]`). 
- The **first element** is the **primary conformer**.
- All subsequent elements are additional decoys/conformers.
- Caliby aggregates Potts parameters across the ensemble to produce the final design.
- By default, `design_ensemble` uses the residue configuration (AA types) of the primary structure for initialization, unless overridden by constraints.



Both this repository and Caliby are still under active development, so please reach out if you have any questions or feature requests! To re-train Caliby, training and dataset preprocessing code should mostly be provided within this repository, but we plan to update in the future with more detailed instructions.

<img src="assets/sampling_gif.gif" alt="Sequence design trajectory" width="600"/>

# Table of Contents
---
**Note: The following sections are the original README from Caliby for reference.**

- [Installation](#installation)
  - [Option 1: Basic `uv` installation](#option-1-basic-uv-installation)
  - [Option 2: Apptainer installation](#option-2-apptainer-installation)
- [Download model weights](#download-model-weights)
- [Usage](#usage)
  - [Sequence design](#sequence-design)
  - [Backbone ensemble generation with Protpardelle-1c](#backbone-ensemble-generation-with-protpardelle-1c)
  - [Ensemble-conditioned sequence design](#ensemble-conditioned-sequence-design)
    - [Sequence design with synthetic Protpardelle-1c ensembles](#sequence-design-with-synthetic-protpardelle-1c-ensembles)
    - [Providing your own ensembles](#providing-your-own-ensembles)
  - [Scoring](#scoring)
  - [Additional sequence design options](#additional-sequence-design-options)
    - [Globally omit certain amino acids](#globally-omit-certain-amino-acids)
    - [Position-wise constraints](#position-wise-constraints)
    - [Specifying symmetry positions](#specifying-symmetry-positions)
  - [Sidechain packing](#sidechain-packing)
- [FAQs](#faqs)
- [License](#license)
- [Citation](#citation)

# Installation
Follow the below instructions for setting up the environment. After you've installed the environment, edit `env_setup.sh` to point to your environment directory, and run `source env_setup.sh` before running any scripts.

## Option 1: Basic `uv` installation
To run the scripts in this repository using `uv`:

Then, run the following commands to install the dependencies:
```
# Clone the repository.
git clone https://github.com/Xiaowei0402/caliby.git
cd caliby

# Create and activate the environment.
ENV_DIR=envs  # or any other directory of your choice
mkdir -p ${ENV_DIR}
uv venv ${ENV_DIR}/caliby -p python3.12
source $ENV_DIR/caliby/bin/activate

# Install the dependencies.
uv pip sync uv_indexes.txt uv.lock -c pins.txt --index-strategy=unsafe-best-match
uv pip install -e .
```

## Option 2: Apptainer installation
If the above workflow does not work for you, and you instead need to run Caliby within an Apptainer container, first set `ENV_DIR` and `IMG` in `build_apptainer.sh` to the directory you want to use for the environment and the image path, respectively. Then, you can run `./build_apptainer.sh` to download the container and install the environment within the container

After the setting up the container and environment, you can run Caliby scripts within the container by wrapping the script in `apptainer exec --nv`. For example, to run `examples/scripts/seq_des_multi.sh` within the container, you can run:
```
IMG=${PWD}/containers/pytorch_24.12.sif
apptainer exec --nv \
  ${IMG} \
  bash -lc '
  source env_setup.sh
  ./examples/scripts/seq_des_multi.sh
'
```

## Download model weights
Weights are not included in the package due to size. You can download them automatically using the wrapper:

```python
from caliby import Caliby

# Download to default location (~/.cache/caliby)
Caliby.download_weights()
```

Alternatively, you can manually run `./download_model_params.sh`, which will download from [Zenodo](https://zenodo.org/records/17263678) and extract into the `model_params/` directory.

We offer two model checkpoints for Caliby:
- `caliby.ckpt`: the default Caliby model trained on all chains in the PDB with 0.3Å Gaussian noise.
- `soluble_caliby.ckpt`: SolubleCaliby, an analog to SolubleMPNN trained by excluding all annotated transmembrane proteins.

# Usage

## Sequence design
To design sequences for a set of PDBs, see `examples/scripts/seq_des_multi.sh`. This script takes in a `input_cfg.pdb_dir` and will design sequences for all PDBs in the directory.

To design sequences for a subset of PDBs within the directory, see `examples/scripts/seq_des_multi_subset.sh`. This script takes in a `input_cfg.pdb_dir` and a `input_cfg.pdb_name_list` (a list of filenames with extensions to use from the directory) and will design sequences for the PDBs specified in the list.

## Backbone ensemble generation with Protpardelle-1c
We found that instead of designing on a static structure, running sequence design on *synthetic ensembles* generated by Protpardelle-1c partial dfifusion produces sequences that are both more diverse and more likely to be predicted by AlphaFold2 to fold into the target structure. To generate ensembles with Protpardelle-1c in a format compatible with Caliby, we have provided a script in `examples/scripts/generate_ensembles.sh`.

For each PDB provided in `input_cfg.pdb_dir`, this script will generate `num_samples_per_pdb` samples per PDB with Protpardelle-1c partial diffusion. For ensemble-conditioned sequence design, we recommend generating at least 32 samples per PDB, but 16 or 8 samples can also give good results.

## Ensemble-conditioned sequence design
### Sequence design with synthetic Protpardelle-1c ensembles
After you've generated ensembles with Protpardelle-1c, you can run ensemble-conditioned sequence design with `examples/scripts/seq_des_multi_ensemble.sh`, which will run ensemble-conditioned sequence design on all ensembles in `input_cfg.conformer_dir`. You can use `examples/scripts/seq_des_multi_ensemble_subset.sh` to run on a subset of the ensembles by providing a `input_cfg.pdb_name_list` file.

### Providing your own ensembles
The Protpardelle-1c ensemble generation script described above will produce a directory structure that is compatible with the `seq_des_multi_ensemble.sh` script, but if you want to provide your own ensembles, you should format your ensembles as described below.

Given a top-level directory passed into the `seq_des_multi_ensemble.sh` script (e.g., `cc95-epoch3490-sampling_partial_diffusion-ss1.0-schurn0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150`), each subdirectory is named `<PDB_KEY>`, representing one ensemble. Inside each ensemble subdirectory, the following files are expected:
  - **Primary conformer**: The original structure file (identified by `<PDB_KEY>.pdb` or `<PDB_KEY>.cif`) serving as the default representative conformer and is always included in the ensemble by default.
  - **Additional conformers**: All other `.pdb` or `.cif` files in the directory are treated as conformer files, which will be ordered by their natural alphabetical order with python's `natsort` library.

The sequence design script takes in a `max_num_conformers` argument, which defaults to 32 and will be used to determine the maximum number of conformers to include in the ensemble. By default, the primary conformer will always be included in the ensemble. Then, we choose additional conformers, in order, until we have `max_num_conformers` conformers in the ensemble or we run out of conformers. We recommend using 32 conformers, but 8 or 16 conformers can also give good results.

## Scoring
All sequence design scripts automatically save the global energy of the sequences they design in the resulting `seq_des_outputs.csv` file. Additionally, Caliby can score sequences on a set of input PDBs via `examples/scripts/score.sh`. This will produce a `score_outputs.csv` file that contains a column `U`, which is the energy computed from the Potts model.

You can also score a sequence against an ensemble of backbones via `examples/scripts/score_ensemble.sh`, where the ensembles should be provided in the same format as described in the previous section. When scoring a sequence against an ensemble, the sequence corresponding to the primary conformer will be scored, and the sequences of the additional conformers will be ignored.

## Additional sequence design options

All sequence design options can be found in `caliby/configs/seq_des/atom_mpnn_inference.yaml`. These options can be overridden from `seq_des_multi.sh` and `seq_des_multi_ensemble.sh` via the `sampling_cfg_overrides` argument using hydra override syntax, e.g. `++sampling_cfg_overrides.omit_aas=["C"]` or `++sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=5000`.

### Globally omit certain amino acids
Caliby can omit certain amino acids from design via the `omit_aas` argument. This should be specified as a list of amino acids to omit, e.g. `["C", "G"]`.

### Position-wise constraints
Caliby also supports applying various constraints on certain residues, specified via the `pos_constraint_csv` argument. This CSV should contain a `pdb_key` column denoting which PDB the constraints should apply to, and a subset of columns in `["fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]`.

| Column name | Format example | Description |
|-------------|----------------|-------------|
| `pdb_key` | `8xhz` | The PDB file stem (filename without extension) to apply the constraints to. For ensemble-conditioned sequence design, `pdb_key` should be the name of the ensemble subdirectory. |
| `fixed_pos_seq` | `A1-100,B1-100` | Sequence positions in the input PDB to condition on so that they remain fixed during design. For ensemble-conditioned design, `fixed_pos_seq` is applied using the primary conformer's sequence. |
| `fixed_pos_scn` | `A1-10,A12,A15-20` | Sidechain positions in the input PDB to condition on so that they remain fixed during design. Note that fixed sidechain positions must be a subset of fixed sequence positions, since it does not make sense to condition on a sidechain without also conditioning on its sequence identity.|
| `fixed_pos_override_seq` | `A26:A,A27:L` | Sequence positions in the input PDB to first override the sequence at, and then condition on. The colon separates the position and the desired amino acid.|
| `pos_restrict_aatype` | `A26:AVG,A27:VG` | Allowed amino acids for certain positions in the input PDB. The colon separates the position and the allowed amino acids.|
| `symmetry_pos` | `A10,B10,C10\|A11,B11,C11` | Symmetry positions for tying sampling across residue positions. The pipe separates groups of positions to sample symmetrically. In the example, A10, B10, and C10 are tied together, and A11, B11, and C11 are tied together. |

If `pos_constraint_csv` is not provided, Caliby will redesign all positions. If a column is not present in the CSV, Caliby will not apply any constraints for that column. If a particular column is included in the CSV, but its value for the `pdb_key` at that row is empty, Caliby will not apply that constraint for that PDB.

For an example constraint CSV file, see `examples/example_data/pos_constraint_csvs/native_pdb_constraints.csv`. For example usage, see `examples/scripts/seq_des_multi_constraints.sh` and `examples/scripts/seq_des_multi_ensemble_constraints.sh`.

In general, residue index positions should be specified by the `label_seq_id` column, *not the `auth_seq_id` column*. To view positions of the sequence using the `label_seq_id` in PyMOL, you can run `set cif_use_auth, off` in PyMOL before loading in the PDB file.

### Specifying symmetry positions
Caliby supports specifying symmetry positions for tying sampling across residue positions. This is specified via the `symmetry_pos` column in the `pos_constraint_csv` file. For an example of a symmetry position CSV file, see `examples/example_data/pos_constraint_csvs/homooligomer_constraints.csv`.

For example usage in sequence design and ensemble generation, see the scripts under the `examples/scripts/homooligomers` directory. We also provide a script to quickly generate a symmetry_pos entry for a homooligomer at `examples/scripts/homooligomers/make_homooligomer_symmetry_pos.sh`, which can be used to print out a symmetry_pos entry by specifying chain IDs and residue ranges.

## Sidechain packing
We plan to add sidechain packing support natively within Caliby in the future. For now, we recommend using Full-Atom MPNN (FAMPNN) for sidechain packing. Please see more details at the official repository for FAMPNN: https://github.com/richardshuai/fampnn.

# FAQs

### I encountered a "Residue index / chain ID mismatch between decoys" error when running ensemble-conditioned sequence design. How do I fix this?

When running ensemble-conditioned sequence design, Caliby requires that the residue indices / chain IDs of the decoys are aligned with the residue indices / chain IDs of the primary conformer. Otherwise, a misalignment could occur when aggregating Potts model parameters across the decoys, which would silently yield poor results. We recommend examining the decoys to ensure all residues line up correctly. If this occurred with a Protpardelle-generated ensemble, this was likely due to inconsistent handling of ignored residues between Protpardelle and Caliby. For this case, we provide a script in `examples/scripts/clean_pdbs.sh` that you can run to clean your PDB file before ensemble generation to ensure that the resulting ensembles are aligned correctly.

# License
Caliby is licensed under the Apache License 2.0. See `LICENSE` for more details.

# Citation
If you use Caliby in your research, you can cite:
```
@article{Shuai2025.09.30.679633,
	author = {Shuai, Richard W. and Lu, Tianyu and Bhatti, Subhang and Kouba, Petr and Huang, Po-Ssu},
	title = {Ensemble-conditioned protein sequence design with Caliby},
	year = {2025},
	doi = {10.1101/2025.09.30.679633},
	journal = {bioRxiv},
	publisher = {Cold Spring Harbor Laboratory},
	url = {https://www.biorxiv.org/content/10.1101/2025.09.30.679633v4},
}
```
If you use Protpardelle-1c for ensemble generation, you can cite:
```
@article{Lu2025.08.18.670959,
  author = {Lu, Tianyu and Shuai, Richard and Kouba, Petr and Li, Zhaoyang and Chen, Yilin and Shirali, Akio and Kim, Jinho and Huang, Po-Ssu},
  title = {Conditional Protein Structure Generation with Protpardelle-1C},
  elocation-id = {2025.08.18.670959},
  year = {2025},
  doi = {10.1101/2025.08.18.670959},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2025/08/19/2025.08.18.670959},
  eprint = {https://www.biorxiv.org/content/early/2025/08/19/2025.08.18.670959.full.pdf},
  journal = {bioRxiv}
}

@article{doi:10.1073/pnas.2311500121,
  author = {Alexander E. Chu  and Jinho Kim  and Lucy Cheng  and  Gina El Nesr  and Minkai Xu  and Richard W. Shuai  and Po-Ssu  Huang },
  title = {An all-atom protein generative model},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {121},
  number = {27},
  pages = {e2311500121},
  year = {2024},
  doi = {10.1073/pnas.2311500121},
  URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2311500121},
  eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2311500121},
}
```
