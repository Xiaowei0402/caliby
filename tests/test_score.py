import pytest
from pathlib import Path
import pandas as pd

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


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_score_single_inmemory_matches_examples():
    csv_path = Path("tests/expected_outputs/score_outputs.csv")
    if not csv_path.exists():
        pytest.skip("Reference score CSV not found")

    df = pd.read_csv(csv_path)
    cal = Caliby(str(CKPT), seed=0, deterministic=True)

    data_dir = Path("tests/test_inputs/native_pdbs")

    grouped = df.groupby("example_id")
    for example_id, group in grouped:
        # find the single input file for this example
        p = None
        for ext in (".pdb", ".cif"):
            cand = data_dir / f"{example_id}{ext}"
            if cand.exists():
                p = cand
                break
        if p is None:
            pytest.skip(f"Input file for {example_id} not found in {data_dir}")

        content = p.read_text()
        out = cal.score(content)

        # compare sequence
        expected_seq = group.iloc[0]["seq"]
        assert out.get("seq", "") == expected_seq

        # compare U numerically
        ref_U = group.iloc[0].get("U", None)
        if pd.notna(ref_U):
            scores = out.get("scores", {})
            out_U = scores.get("U", None) if isinstance(scores, dict) else None
            assert out_U is not None
            assert float(out_U) == pytest.approx(float(ref_U), rel=1e-6, abs=1e-6)
