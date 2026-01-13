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
def test_design_single_inmemory_matches_examples():
    csv_path = Path("examples/outputs/seq_des_multi/seq_des_outputs.csv")
    if not csv_path.exists():
        pytest.skip("Reference seq_des CSV not found")

    df = pd.read_csv(csv_path)
    cal = Caliby(str(CKPT), seed=0, deterministic=True)

    data_dir = Path("examples/example_data/native_pdbs")

    grouped = df.groupby("example_id")

    # Only check the first example_id from the reference CSV
    # (Because random number behavior differs with batched design calls.)
    example_id, group = next(iter(grouped))
    # find the input file for this example
    p = None
    for ext in (".pdb", ".cif"):
        cand = data_dir / f"{example_id}{ext}"
        if cand.exists():
            p = cand
            break
    if p is None:
        pytest.skip(f"Input file for {example_id} not found in {data_dir}")

    content = p.read_text()

    expected_rows = list(group.itertuples(index=False, name=None))
    num_expected = len(expected_rows)

    # Reset seed before each call to ensure per-call determinism
    cal.set_seed(0, deterministic=True)
    results = cal.design(content, num_seqs=num_expected)

    # Extract sequences in order
    got_seqs = [r.get("seq", "") for r in results]
    expected_seqs = list(group["seq"])

    # breakpoint()

    for seq in got_seqs:
        assert seq in expected_seqs, f"Got unexpected sequence for {example_id}: {seq}"
