# NOTE: this is deprecate for now

import pytest
from pathlib import Path
import pandas as pd

# Skip if heavy deps not present
try:
    import torch  # noqa: F401
except Exception:
    pytest.skip("torch missing; skip integration tests", allow_module_level=True)

# Load Caliby implementation directly from repository root to avoid import issues in test envs.
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


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_score_inmemory_matches_examples():
    csv_path = Path("examples/outputs/score_ensemble/score_outputs.csv")
    if not csv_path.exists():
        pytest.skip("Reference score CSV not found")

    df = pd.read_csv(csv_path)
    cal = Caliby(str(CKPT), seed=0, deterministic=True)

    conformer_dirs = Path("examples/outputs/generate_ensembles/cc95-epoch3490-sampling_partial_diffusion-ss1.0-schurn0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150")

    # load conformers into a dict by example_id
    grouped = df.groupby("example_id")
    for example_id, group in grouped:
        # find each directory of conformers
        conformer_dir = conformer_dirs / example_id
        if not conformer_dir.exists():
            pytest.skip(f"Conformer dir for {example_id} not found in {conformer_dirs}")
        # read all conformers
        conformer_structs = []
        for ext in (".pdb", ".cif"):
            # always put main conformer first, main conformer should be named example_id.pdb/cif
            main_path = conformer_dir / f"{example_id}{ext}"
            if main_path.exists():
                conformer_structs.append(main_path.read_text())
            # then add any other conformers
            for p in sorted(conformer_dir.glob(f"*{ext}")):
                if p.name != f"{example_id}{ext}":
                    conformer_structs.append(p.read_text())
        if not conformer_structs:
            pytest.skip(f"No conformers found for {example_id} in {conformer_dir}")

        conformers = {example_id: conformer_structs}

        out = cal.score_ensemble(pdb_to_conformers = conformers)

        row = group.iloc[0]  # there should be only one row per example_id

        # compare sequence
        assert out.get("seq", "") == row["seq"]

        # compare U numerically if present
        ref_U = row.get("U", None)
        if pd.notna(ref_U):
            out_U = None
            scores = out.get("scores", {})
            if isinstance(scores, dict):
                out_U = scores.get("U", None)
            assert out_U is not None
            assert float(out_U) == pytest.approx(float(ref_U), rel=1e-6, abs=1e-6)


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_ensemble_design_inmemory_matches_examples():
    csv_path = Path("examples/outputs/seq_des_multi_ensemble/seq_des_outputs.csv")
    if not csv_path.exists():
        pytest.skip("Reference seq_des CSV not found")

    df = pd.read_csv(csv_path)
    cal = Caliby(str(CKPT), seed=0, deterministic=True)

    conformer_dirs = Path("examples/outputs/generate_ensembles/cc95-epoch3490-sampling_partial_diffusion-ss1.0-schurn0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150")

    # load conformers into a dict by example_id
    grouped = df.groupby("example_id")
    for example_id, group in grouped:
        # find each directory of conformers
        conformer_dir = conformer_dirs / example_id
        if not conformer_dir.exists():
            pytest.skip(f"Conformer dir for {example_id} not found in {conformer_dirs}")
        # read all conformers
        conformer_structs = []
        for ext in (".pdb", ".cif"):
            # always put main conformer first, main conformer should be named example_id.pdb/cif
            main_path = conformer_dir / f"{example_id}{ext}"
            if main_path.exists():
                conformer_structs.append(main_path.read_text())
            # then add any other conformers
            for p in sorted(conformer_dir.glob(f"*{ext}")):
                if p.name != f"{example_id}{ext}":
                    conformer_structs.append(p.read_text())
        if not conformer_structs:
            pytest.skip(f"No conformers found for {example_id} in {conformer_dir}")

        conformers = {example_id: conformer_structs}

        expected_seqs = list(group["seq"])
        num_expected = len(expected_seqs)

        outs = cal.design_ensemble(pdb_to_conformers = conformers, num_seqs_per_pdb=num_expected)

        got_seqs = outs['seq']

        assert got_seqs == expected_seqs, f"Sequences for {example_id} do not match expected outputs: got {got_seqs}, expected {expected_seqs}"
