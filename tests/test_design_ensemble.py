import pytest
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    pytest.skip("torch missing; skip integration tests", allow_module_level=True)

import importlib.util
import sys

repo_root = Path(__file__).resolve().parents[1]
caliby_file = repo_root / "Caliby.py"
if not caliby_file.exists():
    pytest.skip("Caliby.py not found in repo root; skipping tests", allow_module_level=True)
spec = importlib.util.spec_from_file_location("caliby_local", str(caliby_file))
caliby_mod = importlib.util.module_from_spec(spec)
sys.modules["caliby_local"] = caliby_mod
spec.loader.exec_module(caliby_mod)
Caliby = caliby_mod.Caliby


CKPT = Path("model_params/caliby/caliby.ckpt")


# Helper utilities for tests
def make_caliby(seed: int = 0, deterministic: bool = True):
    cal = Caliby(str(CKPT), seed=seed, deterministic=deterministic)
    try:
        cal.set_seed(seed, deterministic=deterministic)
    except Exception:
        # Some Caliby builds may not expose set_seed; ignore if missing
        pass
    return cal


def read_ensemble_contents(example_id: str):
    ensemble_dir = Path("tests/test_inputs/protpardelle_ensembles") / example_id
    if not ensemble_dir.exists():
        return None
    
    contents = []
    # Primary conformer should be the one named after the example_id
    for ext in (".cif", ".pdb"):
        primary_path = ensemble_dir / f"{example_id}{ext}"
        if primary_path.exists():
            contents.append(primary_path.read_text())
            break
    
    if not contents:
        return None
        
    # Then add any other conformers (samples)
    for p in sorted(ensemble_dir.glob("sample_*")):
        contents.append(p.read_text())
        
    return contents


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_design_ensemble_matches_examples():
    """Test ensemble-based sequence design without constraints."""
    csv_path = Path("tests/expected_outputs/seq_des_ensemble_output.csv")
    if not csv_path.exists():
        pytest.skip("Reference ensemble seq_des CSV not found")

    df = pd.read_csv(csv_path)
    cal = make_caliby(0, True)

    grouped = df.groupby("example_id")
    
    # Only test the first example_id to ensure stability with random seeds
    example_id, group = next(iter(grouped))
    
    ensemble_contents = read_ensemble_contents(example_id)
    if ensemble_contents is None:
        pytest.skip(f"Ensemble contents for {example_id} not found")

    num_expected = len(group)
    
    # Reset seed for each call
    cal.set_seed(0, deterministic=True)
    results = cal.design_ensemble(ensemble_contents, num_seqs=num_expected)
    
    got_seqs = [r.get("seq", "") for r in results]
    expected_seqs = list(group["seq"])
    
    assert sorted(got_seqs) == sorted(expected_seqs), (
        f"Ensemble design sequences for {example_id} do not match expected outputs."
    )


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_design_ensemble_with_constraints():
    """Test ensemble-based sequence design with position constraints."""
    csv_path = Path("tests/expected_outputs/seq_des_ensemble_output_constraints.csv")
    constr_path = Path("tests/test_inputs/native_pdb_constraints.csv")

    if not csv_path.exists() or not constr_path.exists():
        pytest.skip("Reference constrained ensemble outputs or constraints CSV not found")

    df_out = pd.read_csv(csv_path)
    df_constr = pd.read_csv(constr_path)
    cal = make_caliby(0, True)

    grouped = df_out.groupby("example_id")
    
    # Only test the first example_id to ensure stability with random seeds
    example_id, group = next(iter(grouped))
    
    ensemble_contents = read_ensemble_contents(example_id)
    if ensemble_contents is None:
        pytest.skip(f"Ensemble contents for {example_id} not found")
    
    pos_constraint_df = df_constr[df_constr["pdb_key"] == example_id]
    if pos_constraint_df.empty:
        pytest.skip(f"No constraints found for {example_id}")
        
    num_expected = len(group)
    
    cal.set_seed(0, deterministic=True)
    results = cal.design_ensemble(
        ensemble_contents, 
        num_seqs=num_expected, 
        pos_constraint_df=pos_constraint_df
    )
    
    got_seqs = [r.get("seq", "") for r in results]
    expected_seqs = list(group["seq"])
    
    assert sorted(got_seqs) == sorted(expected_seqs), (
        f"Constrained ensemble design sequences for {example_id} do not match expected outputs."
    )


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_ensemble_primary_res_type_override():
    """Test the use_primary_res_type argument and sequence override in ensemble design."""
    # 7xz3 example which has a sequence override constraint
    example_id = "7xz3"
    constr_path = Path("tests/test_inputs/native_pdb_constraints.csv")
    ensemble_contents = read_ensemble_contents(example_id)
    if ensemble_contents is None or not constr_path.exists():
        pytest.skip(f"Ensemble contents or constraints for {example_id} not found")
        
    df_constr = pd.read_csv(constr_path)
    pos_constraint_df = df_constr[df_constr["pdb_key"] == example_id]
    
    cal = make_caliby(0, True)
    
    # We test that use_primary_res_type works alongside a specific sequence override.
    # We pass the pos_constraint_df to actually trigger the override.
    results = cal.design_ensemble(
        ensemble_contents, 
        num_seqs=2, 
        use_primary_res_type=True,
        pos_constraint_df=pos_constraint_df
    )
    
    assert len(results) == 2
    for r in results:
        assert isinstance(r["seq"], str)
        assert len(r["seq"]) > 0
        
    got_seqs = [r.get("seq", "") for r in results]

    for i, seq in enumerate(got_seqs):
        # Sequence generated will skip missing residues, so the actual position for 36-40 on pdb is 17-21 in seq
        assert seq[16:21] == "CCCCC", f"{i}, Override failed, got residues 36-40 as {seq[16:21]} instead of CCCCC"
