import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union, List, Dict

import torch
import torch.nn.functional as F
import hydra
import pandas as pd
import numpy as np
import random
from omegaconf import OmegaConf, DictConfig

from atomworks.io.parser import parse as aw_parse
from atomworks.io.utils import non_rcsb
from atomworks.io.utils.io_utils import to_cif_string

from caliby.checkpoint_utils import get_cfg_from_ckpt
from caliby.data.data import to
from caliby.data.datasets.atomworks_sd_dataset import sd_collator
from caliby.data.transform.preprocess import preprocess_transform
from caliby.data.transform.sd_featurizer import sd_featurizer
from caliby.model.seq_denoiser.lit_sd_model import LitSeqDenoiser

from atomworks.ml.utils.token import spread_token_wise
import caliby.data.const as const

from caliby.eval.eval_utils import seq_des_utils as _seq_des_utils


try:
    import lightning as _lightning
except Exception:
    _lightning = None


class Caliby:
    def __init__(self, checkpoint_path: str, device: str = None, sampling_overrides: dict = None, seed: Optional[int] = None, deterministic: bool = True):
        """
        Initialize the Caliby model.

        Args:
            checkpoint_path: Path to the model checkpoint (.ckpt file).
            device: Device to run on ("cuda", "cpu", etc.). If None, auto-detect.
            sampling_overrides: Dictionary of overrides for sampling configuration.
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set global RNG seed if provided
        if seed is not None:
            self.set_seed(seed, deterministic=deterministic)

        # Load model and configs
        self.model, self.data_cfg, self.sampling_cfg = self._load_model(checkpoint_path, sampling_overrides)
        self.model.eval()
        self.model.to(self.device)

    def set_seed(self, seed: int, deterministic: bool = True) -> None:
        """Set global random seeds for Python, NumPy, PyTorch, and Lightning (if available).

        Args:
            seed: integer seed to set.
            deterministic: whether to enable deterministic cuDNN behavior.
        """
        # Python random
        random.seed(seed)
        # NumPy
        np.random.seed(seed)
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Try to use Lightning's seeding if available
        # Use lightning if available (imported at module level)
        try:
            if _lightning is not None:
                _lightning.seed_everything(seed)
        except Exception:
            # fallback: already seeded python/numpy/torch
            pass

        # Set cuDNN deterministic flags to improve reproducibility
        try:
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = not bool(deterministic)
        except Exception:
            pass

        # Record
        self.seed = seed
        self.deterministic = bool(deterministic)

    def _load_model(self, ckpt_path: str, overrides: dict = None):
        """Load model, data config, and sampling config from checkpoint."""
        # Load model using map_location to ensure it goes to the right device directly or CPU first
        lit_sd_model = LitSeqDenoiser.load_from_checkpoint(ckpt_path, map_location=self.device).eval()
        model_cfg, _ = get_cfg_from_ckpt(ckpt_path)
        
        # Instantiate data config
        # Handle cases where data_cfg might depend on other configs or be missing
        if hasattr(model_cfg, 'data'):
            data_cfg = hydra.utils.instantiate(model_cfg.data)
        else:
            # Fallback handling if data config is not in model config
            data_cfg = DictConfig({}) 

        # Load sampling config
        # Try to locate the default sampling config file relative to the package
        base_dir = Path(__file__).resolve().parent
        # Fixed path based on file search
        default_sampling_cfg_path = base_dir / "caliby/configs/seq_des/atom_mpnn_inference.yaml"
        
        if default_sampling_cfg_path.exists():
             sampling_cfg = OmegaConf.load(default_sampling_cfg_path)
        else:
            # Fallback with essential defaults
            sampling_cfg = OmegaConf.create({
                "num_seqs_per_pdb": 1,
                "temperature": 0.1,
                "batch_size": 1,
                "num_workers": 0,
                "verbose": False,
                # These are often used in sampling masks
                "ensemble_ignore_res_idx_mismatch": False,
                "omit_aas": None, 
                "gaussian_conformers_cfg": {
                    "n_conformers": 0,
                    "noise_std": 0.0
                }
            })

        if overrides:
            sampling_cfg = OmegaConf.merge(sampling_cfg, OmegaConf.create(overrides))
            
        return lit_sd_model.model, data_cfg, sampling_cfg

    def _process_input(self, pdb_content: str, suffix: str = None, example_id: str = "input_pdb") -> dict[str, Any]:
        """Parse, preprocess, and featurize a PDB/CIF string."""

        # Auto-detect file type if not provided
        file_type = None
        if suffix is not None:
            file_type = suffix.lstrip(".")

        if file_type is None:
            if pdb_content.strip().startswith("data_"):
                file_type = "cif"
            elif "ATOM " in pdb_content or "HETATM" in pdb_content:
                file_type = "pdb"
            else:
                # Default to cif if ambiguous
                file_type = "cif"

        # Setup CIF parser args
        if hasattr(self.data_cfg, "cif_parser_args"):
            cif_parser_args = OmegaConf.to_container(self.data_cfg.cif_parser_args, resolve=True)
        else:
            # Default args if not found in config
            cif_parser_args = {
                "add_missing_atoms": True,
                "remove_waters": True,
                "remove_ccds": [],
                "fix_ligands_at_symmetry_centers": True,
                "fix_arginines": True,
                "convert_mse_to_met": True,
                "hydrogen_policy": "remove",
            }

        # aw_parse usually returns a dict with 'assemblies'
        # We assume transformation_id "1" is standard
        transformation_id = "1"
        cif_parser_args["build_assembly"] = [transformation_id]

        # Use StringIO to avoid temp files.
        # Note: atomworks.io.parser._parse_from_pdb has a bug where it calls Path(filename)
        # on buffers. We temporarily monkeypatch Path in that module to handle buffers safely.
        import atomworks.io.parser as aw_parser_mod
        original_path = aw_parser_mod.Path

        def buffer_friendly_path(x):
            if isinstance(x, (io.StringIO, io.BytesIO)):
                return original_path(f"input.{file_type}")
            return original_path(x)

        aw_parser_mod.Path = buffer_friendly_path

        try:
            # Pass content directly via StringIO buffer
            input_data = aw_parse(io.StringIO(pdb_content), file_type=file_type, **cif_parser_args)
        except Exception as e:
            # If parsing fails, it might be due to format mismatch or empty content
            raise ValueError(f"Failed to parse input structure: {e}")
        finally:
            aw_parser_mod.Path = original_path

        if "assemblies" not in input_data or transformation_id not in input_data["assemblies"]:
            raise ValueError("Parsed data does not contain expected assembly information.")

        atom_array_from_cif = input_data["assemblies"][transformation_id][0]

        # Preprocess
        pipeline = preprocess_transform()
        example = pipeline(
            data={
                "example_id": example_id,
                "atom_array": atom_array_from_cif,
                "chain_info": input_data["chain_info"],
            }
        )

        # Featurize
        featurizer = sd_featurizer()
        example = featurizer(example)

        return example

    def design(self,
               pdb_content: str,
               num_seqs: int = 1,
               temperature: float = 0.1,
               pos_constraint_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
        """
        Design sequences for a given PDB structure.

        Args:
            pdb_content: content of the PDB/CIF file.
            num_seqs: number of sequences to generate.
            temperature: sampling temperature.
            pos_constraint_df: optional constraints.

        Returns:
            List of dictionaries containing 'seq', 'pdb_string', 'scores'.
        """
        # Update sampling config locally
        sampling_cfg = self.sampling_cfg.copy()
        sampling_cfg.num_seqs_per_pdb = num_seqs
        sampling_cfg.temperature = temperature
        
        # Prepare batch and apply constraints
        batch = _prepare_batch(self, [pdb_content])
        batch, sampling_inputs = _finalize_constraints(batch, sampling_cfg, pos_constraint_df)
        
        # Run sampling
        with torch.no_grad():
            id_to_atom_arrays, id_to_aux = self.model.sample(batch, sampling_inputs=sampling_inputs)
        
        return _format_design_results(id_to_atom_arrays, id_to_aux)

    def design_ensemble(self,
                        pdb_contents: list[str],
                        num_seqs: int = 1,
                        temperature: float = 0.1,
                        pos_constraint_df: pd.DataFrame | None = None,
                        use_primary_res_type: bool = True) -> list[dict[str, Any]]:
        """
        Design sequences for a given PDB structure ensemble using in-memory inputs.

        Args:
            pdb_contents: List of PDB/CIF strings representing the ensemble. 
                The first string in the list is treated as the primary conformer 
                and used as the template for output structures and residue alignment.
            num_seqs: number of sequences to generate.
            temperature: sampling temperature.
            pos_constraint_df: optional constraints.
            use_primary_res_type: use res_type from primary structure (the first string in pdb_contents).

        Returns:
            List of dictionaries containing 'seq', 'pdb_string', 'scores'.
        """
        # Update sampling config locally
        sampling_cfg = self.sampling_cfg.copy()
        sampling_cfg.num_seqs_per_pdb = num_seqs
        sampling_cfg.temperature = temperature
        
        # Prepare batch
        batch = _prepare_batch(self, pdb_contents)
        
        # Apply ensemble specific logic
        ignore_mismatch = sampling_cfg.get("ensemble_ignore_res_idx_mismatch", False)
        batch = _apply_ensemble_logic(batch, use_primary_res_type, ignore_mismatch, self.device)
        
        # Apply constraints
        batch, sampling_inputs = _finalize_constraints(batch, sampling_cfg, pos_constraint_df)
        
        # Run sampling
        with torch.no_grad():
            id_to_atom_arrays, id_to_aux = self.model.sample(batch, sampling_inputs=sampling_inputs)
        
        return _format_design_results(id_to_atom_arrays, id_to_aux)

    def score(self, pdb_content: str) -> dict[str, Any]:
        """Score the sequence in the provided PDB/CIF string."""
        # Prepare batch and apply constraints (retrieving sampling_inputs container)
        batch = _prepare_batch(self, [pdb_content])
        batch, sampling_inputs = _finalize_constraints(batch, self.sampling_cfg, None)
        
        with torch.no_grad():
            id_to_aux = self.model.score_samples(batch, sampling_inputs=sampling_inputs)
            
        return _format_score_results(id_to_aux)

    def score_ensemble(self, pdb_contents: list[str]) -> dict[str, Any]:
        """
        Score a sequence using Potts parameters computed from an ensemble of structures.
        The sequence from the first structure is used for scoring.

        Args:
            pdb_contents: List of PDB/CIF strings representing the ensemble.
                The first string in the list is treated as the primary conformer
                and used to provide the sequence for scoring.

        Returns:
            Dictionary containing 'seq' and 'scores'.
        """
        # Prepare batch
        batch = _prepare_batch(self, pdb_contents)

        # Apply ensemble specific logic
        ignore_mismatch = self.sampling_cfg.get("ensemble_ignore_res_idx_mismatch", False)
        # Use primary res types since we are scoring the sequence from the primary structure
        batch = _apply_ensemble_logic(batch, use_primary_res_type=True, ignore_mismatch=ignore_mismatch, device=self.device)

        # Finalize inputs
        batch, sampling_inputs = _finalize_constraints(batch, self.sampling_cfg, None)

        with torch.no_grad():
            id_to_aux = self.model.score_samples(batch, sampling_inputs=sampling_inputs)

        return _format_score_results(id_to_aux)

    

    def _initialize_sampling_masks(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Delegate sampling mask initialization to shared utility."""
        if _seq_des_utils is not None:
            return _seq_des_utils.initialize_sampling_masks(batch)
        else:
            # Fallback: try to implement minimal behavior inline
            standard_prot_mask = batch["is_protein"] & ~batch["is_atomized"]
            batch["seq_cond_mask"] = torch.zeros_like(batch["token_pad_mask"])
            batch["seq_cond_mask"] = torch.where(
                standard_prot_mask, torch.zeros_like(batch["seq_cond_mask"]), batch["token_resolved_mask"]
            )

            batch["atom_cond_mask"] = batch["prot_bb_atom_mask"]
            atomwise_standard_prot_mask = (
                torch.gather(standard_prot_mask, dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"]
            )
            batch["atom_cond_mask"] = torch.where(
                atomwise_standard_prot_mask.bool(), batch["atom_cond_mask"], batch["atom_resolved_mask"]
            )

            return batch


def _prepare_batch(caliby, pdb_contents: list[str], example_id: str = "input_pdb"):
    """
    Common logic to process PDB contents into a featurized batch.
    """
    examples = [caliby._process_input(content, example_id=example_id) for content in pdb_contents]
    batch = sd_collator(examples)
    batch = to(batch, caliby.device)
    batch = caliby._initialize_sampling_masks(batch)
    return batch


def _apply_ensemble_logic(batch, use_primary_res_type, ignore_mismatch, device):
    """
    Logic for ensemble inputs: tie samples and align residue types if needed.
    """
    # Tie all samples together
    batch["tied_sampling_ids"] = torch.zeros(len(batch["example_id"]), device=device, dtype=torch.long)

    # Use res_type from primary structure (the first in the ensemble)
    if use_primary_res_type:
        # We use repeat instead of expand to allow independent (though usually identical) 
        # modifications during the constraint application phase.
        repeat_dims = [len(batch["example_id"])] + [1] * (batch["restype"].ndim - 1)
        batch["restype"] = batch["restype"][0:1].repeat(*repeat_dims)
        
        # Update atom array names for all other conformers
        for i in range(1, len(batch["atom_array"])):
            atomwise_resnames = spread_token_wise(
                batch["atom_array"][i],
                const.AF3_ENCODING.idx_to_token[batch["restype"][0].argmax(dim=-1).cpu().numpy()],
            )
            batch["atom_array"][i].set_annotation("res_name", atomwise_resnames)

    # Validate alignment
    if not ignore_mismatch:
        if not (batch["residue_index"] == batch["residue_index"][0]).all().item():
            raise ValueError("Residue index mismatch between decoys.")
        if not (batch["asym_id"] == batch["asym_id"][0]).all().item():
            raise ValueError("Chain ID mismatch between decoys.")
    return batch


def _finalize_constraints(batch, sampling_cfg, pos_constraint_df):
    """
    Common logic to parse and apply constraints into sampling_inputs.
    """
    sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)

    if pos_constraint_df is not None:
        # Normalize index if pdb_key is provided
        if "pdb_key" in pos_constraint_df.columns:
            pos_constraint_df = pos_constraint_df.set_index("pdb_key")

        # For single-target in-memory design, if the dataframe has one row, 
        # map it to the current batch's example_id to ensure constraints are applied.
        if pos_constraint_df.shape[0] == 1:
            example_id_val = batch["example_id"][0]
            if pos_constraint_df.index[0] != example_id_val:
                pos_constraint_df = pos_constraint_df.copy()
                pos_constraint_df.index = [example_id_val]

        if _seq_des_utils is not None:
            batch = _seq_des_utils.parse_fixed_pos_info(batch, pos_constraint_df)
            sampling_inputs["pos_restrict_aatype"] = _seq_des_utils.parse_pos_restrict_aatype_info(
                batch, pos_constraint_df
            )
            sampling_inputs["symmetry_pos"] = _seq_des_utils.parse_symmetry_pos_info(batch, pos_constraint_df)

    # Ensure optional keys are present
    sampling_inputs.setdefault("pos_restrict_aatype", None)
    sampling_inputs.setdefault("symmetry_pos", None)

    return batch, sampling_inputs


def _format_design_results(id_to_atom_arrays, id_to_aux, example_id="input_pdb"):
    """
    Helper to format design results into standard list of dicts.
    """
    results = []
    if example_id in id_to_atom_arrays:
        atom_arrays = id_to_atom_arrays[example_id]
        aux = id_to_aux[example_id]

        for si, atom_array in enumerate(atom_arrays):
            # Sequence
            chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_array)
            seq = ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())

            # PDB String
            pdb_str = to_cif_string(atom_array, include_nan_coords=False)

            # Scores
            scores = aux[si] if isinstance(aux, list) else aux
            current_score = scores if isinstance(scores, dict) else {}

            results.append({"seq": seq, "pdb_string": pdb_str, "scores": current_score})
    return results


def _format_score_results(id_to_aux, example_id="input_pdb"):
    """
    Helper to format score results into standard dict.
    """
    if example_id in id_to_aux:
        aux = id_to_aux[example_id]

        # Get sequence
        chain_info = non_rcsb.initialize_chain_info_from_atom_array(aux["atom_array"])
        seq = ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())

        return {"seq": seq, "scores": aux}
    return {}
