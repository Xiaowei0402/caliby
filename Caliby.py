import tempfile
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
from atomworks.ml.utils.token import spread_token_wise, apply_token_wise, get_token_starts

from caliby.checkpoint_utils import get_cfg_from_ckpt
from caliby.data.data import to
from caliby.data.datasets.atomworks_sd_dataset import sd_collator
from caliby.data.transform.preprocess import preprocess_transform
from caliby.data.transform.sd_featurizer import sd_featurizer
from caliby.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
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
        
        # Auto-detect suffix if not provided
        if suffix is None:
            if pdb_content.strip().startswith("data_"):
                suffix = ".cif"
            elif "ATOM " in pdb_content or "HETATM" in pdb_content:
                suffix = ".pdb"
            else:
                # Default to cif if ambiguous or maybe fasta (not supported here yet)
                suffix = ".cif"
            
        # Write pdb_content to temp file for aw_parse
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix) as tmp_file:
            tmp_file.write(pdb_content)
            tmp_file.flush()
            
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
            
            try:
                input_data = aw_parse(tmp_file.name, **cif_parser_args)
            except Exception as e:
                # If parsing fails, it might be due to format mismatch or empty content
                raise ValueError(f"Failed to parse input structure: {e}")

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

        Returns:
            List of dictionaries containing 'seq', 'pdb_string', 'scores'.
        """
        # Update sampling config locally
        sampling_cfg = self.sampling_cfg.copy()
        sampling_cfg.num_seqs_per_pdb = num_seqs
        sampling_cfg.temperature = temperature
        
        # Process input
        example = self._process_input(pdb_content)
        
        # Create batch 
        batch = sd_collator([example])
        batch = to(batch, self.device)
        
        # Initialize masks
        batch = self._initialize_sampling_masks(batch)
        
        # Finalize constraints: format inputs into a canonical pos_constraint_df and use canonical parsers
        sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)

        # If user provided a pos_constraint_df (CSV-like DataFrame), normalize its index and use canonical parsers.
        seq_des_utils = _seq_des_utils
        if pos_constraint_df is not None:
            # If CSV style with column 'pdb_key', normalize index.
            if "pdb_key" in pos_constraint_df.columns:
                # If user supplied a filtered one-row DataFrame (e.g., df[df['pdb_key']==id]),
                # the index may be a RangeIndex; canonical parsers expect the example_id in
                # batch["example_id"] to match the DataFrame index. For in-memory calls we
                # remap the single-row DataFrame's index to the batch example id so parsers find it.
                pos_constraint_df = pos_constraint_df.set_index("pdb_key")


                # If this is a single-row DataFrame, remap its index to the batch example id
                if pos_constraint_df.shape[0] == 1:
                    example_id_val = batch["example_id"][0]
                    pos_constraint_df.index = [example_id_val]

            if seq_des_utils is not None:
                batch = seq_des_utils.parse_fixed_pos_info(batch, pos_constraint_df)
                sampling_inputs["pos_restrict_aatype"] = seq_des_utils.parse_pos_restrict_aatype_info(batch, pos_constraint_df)
                sampling_inputs["symmetry_pos"] = seq_des_utils.parse_symmetry_pos_info(batch, pos_constraint_df)
            else:
                sampling_inputs.setdefault("pos_restrict_aatype", None)
                sampling_inputs.setdefault("symmetry_pos", None)
        else:

            sampling_inputs.setdefault("pos_restrict_aatype", None)
            sampling_inputs.setdefault("symmetry_pos", None)
        
        # Run sampling
        with torch.no_grad():
            id_to_atom_arrays, id_to_aux = self.model.sample(batch, sampling_inputs=sampling_inputs)
        
        # Format output
        results = []
        example_id = "input_pdb"
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
                # aux might be list of dicts or dict of lists, verifying from seq_des_utils:
                # outputs["U"].append(aux[si]["U"]) implies aux is a list of dicts
                
                current_score = scores if isinstance(scores, dict) else {}
                
                results.append({
                    "seq": seq,
                    "pdb_string": pdb_str,
                    "scores": current_score
                })
                
        return results

    def score(self, pdb_content: str) -> dict[str, Any]:
        """Score the sequence in the provided PDB/CIF string."""
        example = self._process_input(pdb_content)
        batch = sd_collator([example])
        batch = to(batch, self.device)
        
        batch = self._initialize_sampling_masks(batch)
        
        sampling_inputs = OmegaConf.to_container(self.sampling_cfg, resolve=True)
        
        with torch.no_grad():
            id_to_aux = self.model.score_samples(batch, sampling_inputs=sampling_inputs)
            
        # Format output
        example_id = "input_pdb"
        if example_id in id_to_aux:
            aux = id_to_aux[example_id]
            
            # Get sequence
            chain_info = non_rcsb.initialize_chain_info_from_atom_array(aux["atom_array"])
            seq = ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
            
            return {
                "seq": seq,
                "scores": aux
            }
        return {}

    
    # def _apply_constraints(self, batch, fixed_positions: dict[str, list[int]] = None, fixed_sidechains: dict[str, list[int]] = None, override_sequence: dict[str, str] = None):
    #     """
    #     Apply fixed positions and override sequence constraints to the batch.
    #     """
    #     # Assuming batch specific for one example duplicated or single
    #     # Here we only handle the single example case effectively (B=1)
        
    #     # Get atom array (assuming single example)
    #     atom_array = batch["atom_array"][0] # This is a reference, modifying it affects batch if linked
        
    #     # Get token mapping
    #     token_starts = get_token_starts(atom_array)
    #     token_chain_ids = atom_array.chain_id[token_starts]

    #     # Validate fixed_sidechains is subset of fixed_positions if both provided
    #     if fixed_sidechains and not fixed_positions:
    #         raise ValueError("fixed_sidechains provided but fixed_positions is missing; sidechains must be a subset of fixed_positions")

    #     if fixed_sidechains and fixed_positions:
    #         for chain, sc_list in fixed_sidechains.items():
    #             if chain not in fixed_positions:
    #                 raise ValueError(f"fixed_sidechains contains chain {chain} not present in fixed_positions")
    #             missing = set(sc_list) - set(fixed_positions.get(chain, []))
    #             if missing:
    #                 raise ValueError(f"fixed_sidechains for chain {chain} contains residues not in fixed_positions: {missing}")

    #     # 1. Override sequence and positional restrictions
    #     # Use canonical parsers from seq_des_utils when possible to ensure identical semantics.
    #     override_entries = []
    #     restrict_entries = []

    #     # Normalize incoming override_sequence which may be provided as per-chain dicts or legacy full-sequence strings
    #     if override_sequence:
    #         for chain_id, spec in override_sequence.items():
    #             # legacy full-sequence string for a chain
    #             if isinstance(spec, str) and not isinstance(spec, dict):
    #                 # treat as full-sequence override for this chain (legacy behavior)
    #                 chain_indices = np.where(token_chain_ids == chain_id)[0]
    #                 if len(chain_indices) == 0:
    #                     continue
    #                 if len(spec) != len(chain_indices):
    #                     raise ValueError(f"Override sequence length for chain {chain_id} is {len(spec)}, "
    #                                      f"but structure has {len(chain_indices)} residues.")

    #                 encoded_seq = const.AF3_ENCODING.encode_aa_seq(spec)
    #                 encoded_tensor = torch.tensor(encoded_seq, device=self.device)
    #                 one_hot = F.one_hot(encoded_tensor, num_classes=const.AF3_ENCODING.n_tokens).float()
    #                 batch["restype"][0, chain_indices] = one_hot
    #             elif isinstance(spec, dict):
    #                 # per-position specs: convert to canonical strings
    #                 for pos_key, val in spec.items():
    #                     try:
    #                         pos_i = int(pos_key)
    #                     except Exception:
    #                         continue

    #                     if isinstance(val, str) and len(val) == 1:
    #                         override_entries.append(f"{chain_id}{pos_i}:{val}")
    #                     else:
    #                         # treat as soft restriction (allowed aatypes)
    #                         if isinstance(val, (list, tuple)):
    #                             letters = "".join([str(v) for v in val])
    #                         else:
    #                             letters = str(val)
    #                         restrict_entries.append(f"{chain_id}{pos_i}:{letters}")

    #     # Note: any positional restrictions passed via the design() call are merged into
    #     # `override_sequence` by the caller; they will be present in `restrict_entries` above.

    #     # Use module-level canonical parsers (imported at module load time)
    #     seq_des_utils = _seq_des_utils

    #     # If canonical utilities are available, construct a small DataFrame for this single-example batch
    #     # and let the canonical parsers populate batch masks and restype exactly as the file-backed pipeline.
    #     if seq_des_utils is not None and (override_entries or restrict_entries):
    #         example_id = batch["example_id"][0]
    #         row = {}
    #         row["fixed_pos_override_seq"] = ",".join(override_entries) if override_entries else np.nan
    #         row["pos_restrict_aatype"] = ",".join(restrict_entries) if restrict_entries else np.nan

    #         pos_df = pd.DataFrame([row], index=[example_id])

    #         # Let canonical parser update seq_cond_mask, atom_cond_mask, and restype overrides
    #         batch = seq_des_utils.parse_fixed_pos_info(batch, pos_df)

    #         # Let canonical parser produce pos_restrict_aatype masks
    #         pr = seq_des_utils.parse_pos_restrict_aatype_info(batch, pos_df)
    #         if pr is not None:
    #             batch["pos_restrict_aatype"] = pr
            
    #     # 2. Fixed positions
    #     if fixed_positions:
    #         token_res_ids = atom_array.res_id[token_starts]
            
    #         # Create set of fixed (chain, res_id) tuples for O(1) lookup
    #         fixed_set = set()
    #         for chain, res_list in fixed_positions.items():
    #             for res_id in res_list:
    #                 fixed_set.add((chain, res_id))
            
    #         # Identify which tokens are fixed
    #         num_tokens = int(batch["token_pad_mask"][0].sum())
    #         mask_indices = []
            
    #         for k in range(num_tokens):
    #             chain = token_chain_ids[k]
    #             res_id = token_res_ids[k]
    #             if (chain, res_id) in fixed_set:
    #                 mask_indices.append(k)
            
    #         if mask_indices:
    #             # Set seq_cond_mask to 1 at these positions
    #             batch["seq_cond_mask"][0, mask_indices] = 1.0

    #     # 3. Fixed sidechains: fix atom positions for non-backbone atoms at those residues
    #     if fixed_sidechains:
    #         token_res_ids = atom_array.res_id[token_starts]

    #         # build set for quick lookup
    #         sc_set = set()
    #         for chain, res_list in fixed_sidechains.items():
    #             for res_id in res_list:
    #                 sc_set.add((chain, res_id))

    #         # identify token indices corresponding to these residues
    #         num_tokens = int(batch["token_pad_mask"][0].sum())
    #         token_indices_to_fix = []
    #         for k in range(num_tokens):
    #             chain = token_chain_ids[k]
    #             res_id = token_res_ids[k]
    #             if (chain, res_id) in sc_set:
    #                 token_indices_to_fix.append(k)

    #         if token_indices_to_fix:
    #             atom_to_token = batch["atom_to_token_map"][0]
    #             prot_bb = batch.get("prot_bb_atom_mask")
    #             atom_pad = batch.get("atom_pad_mask")
    #             if "atom_cond_mask" not in batch:
    #                 batch["atom_cond_mask"] = torch.zeros_like(batch["atom_pad_mask"]) 

    #             atom_indices = []
    #             for ai in range(atom_to_token.shape[0]):
    #                 if atom_pad is not None:
    #                     if not atom_pad[0, ai]:
    #                         continue
    #                 tok = int(atom_to_token[ai].item())
    #                 if tok in token_indices_to_fix:
    #                     is_bb = bool(prot_bb[0, ai]) if prot_bb is not None else False
    #                     if not is_bb:
    #                         atom_indices.append(ai)

    #             if atom_indices:
    #                 batch["atom_cond_mask"][0, atom_indices] = 1.0

    #     return batch

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
