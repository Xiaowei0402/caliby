# Plan for Refactoring Caliby

The goal is to create a new `Caliby` class that supports in-memory sequence design and scoring, minimizing file I/O.

## 1. Create `caliby/inference.py`
This file will contain the new `Caliby` class.

### Class Structure: `Caliby`
- **`__init__(self, checkpoint_path: str, device: str = None)`**:
  - Initializes the device (CUDA/CPU).
  - Loads the config and the PyTorch Lightning model (`LitSeqDenoiser`) from the checkpoint.
  - Instantiates the data config and sampling config.
  - Sets the model to eval mode.

- **`design(self, pdb_content: str, num_seqs: int = 1, ...)`**:
  - Accepts PDB content as a string.
  - Parses the PDB content (using a temporary file if necessary for `aw_parse`).
  - Preprocesses and featurizes the structure.
  - Sets up the batch (including creating masks).
  - Runs the model's `sample` method.
  - Returns a list of results (designed sequences, CIF strings of designed structures, scores).

- **`score(self, pdb_content: str, ...)`**:
  - Accepts PDB content as a string.
  - Parses, preprocesses, and featurizes.
  - Runs the model's `score_samples` method.
  - Returns scores.

- **Helper Methods (Private)**:
  - `_load_model_from_checkpoint`: Logic from `get_seq_des_model`.
  - `_process_input(pdb_content)`: Handles parsing (via tempfile + `aw_parse`), preprocessing (`preprocess_transform`), and featurizing (`sd_featurizer`).
  - `_prepare_batch(examples)`: Collates and moves to device.
  - `_initialize_masks(batch)`: Functionality from `initialize_sampling_masks`.

## 2. Refactoring Dependencies
To make `caliby/inference.py` self-contained or cleanly dependent, I will likely need to:
- Import logic from `caliby.eval.eval_utils.seq_des_utils` or copy relevant parts if they represent specific "script" logic vs "library" logic.
- The `preprocess_pdb` function in `seq_des_utils.py` currently couples file reading and preprocessing. I will reimplement a version that works on the parsed data.

## 3. Implementation Details
- **Input Parsing**: Since `aw_parse` likely requires a file path, I will use `tempfile.NamedTemporaryFile` to write the `pdb_content` string to a temporary file, pass that path to `aw_parse`, and then clean up.
- **Data Pipeline**: Replicate the pipeline: `aw_parse` -> `preprocess_transform` -> `sd_featurizer`.
- **Output**: instead of writing CIFs to disk, use `to_cif_string` and return the string.

## 4. Verification
- Create a test script `test_caliby_inference.py`.
- Load a sample PDB.
- Instantiate `Caliby`.
- Run `design` and print output sequences.
- Run `score` and print scores.

