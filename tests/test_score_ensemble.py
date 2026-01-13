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
def test_score_ensemble_matches_examples():
    """Test ensemble-based sequence scoring."""
    # Use the example output scores which are calibrated for ensembles
    csv_path = Path("tests/expected_outputs/score_ensemble_outputs.csv")

    df = pd.read_csv(csv_path)
    cal = make_caliby(0, True)

    grouped = df.groupby("example_id")
    
    for example_id, group in grouped:
        ensemble_contents = read_ensemble_contents(example_id)
        if ensemble_contents is None:
            # Maybe it's missing in test_inputs but present in examples?
            # For now, if we can't find it, we skip this example
            continue

        # Score ensemble
        out = cal.score_ensemble(ensemble_contents)
        
        # compare sequence
        expected_seq = group.iloc[0]["seq"]
        assert out.get("seq", "") == expected_seq

        # compare U numerically
        ref_U = group.iloc[0].get("U", None)
        if pd.notna(ref_U):
            scores = out.get("scores", {})
            out_U = scores.get("U", None) if isinstance(scores, dict) else None
            assert out_U is not None
            assert float(out_U) == pytest.approx(float(ref_U), rel=1e-5, abs=1e-5)

if __name__ == "__main__":
    # For manual debugging
    if CKPT.exists():
        test_score_ensemble_matches_examples()
    else:
        print("CKPT not found")
