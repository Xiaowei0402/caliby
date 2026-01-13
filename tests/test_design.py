import pytest
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    pytest.skip("torch missing; skip integration tests", allow_module_level=True)

from caliby.data.datasets.atomworks_sd_dataset import sd_collator
from caliby.data.data import to
from atomworks.ml.utils.token import get_token_starts
from caliby import data as _data  # to access helpers if needed
import caliby.data.const as _const


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


def read_pdb_content(example_id: str):
    data_dir = Path("tests/test_inputs/native_pdbs")
    for ext in (".pdb", ".cif"):
        cand = data_dir / f"{example_id}{ext}"
        if cand.exists():
            return cand.read_text()
    return None


def get_pos_constraint_df_for(pdb_key: str):
    constr_path = Path("tests/test_inputs/native_pdb_constraints.csv")
    if not constr_path.exists():
        return None
    df_constr = pd.read_csv(constr_path)
    df_sub = df_constr[df_constr["pdb_key"] == pdb_key]
    if df_sub.empty:
        return None
    return df_sub

# unit test for sequence design
@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_design_single_inmemory_matches_examples():
    csv_path = Path("tests/expected_outputs/seq_des_output.csv")
    if not csv_path.exists():
        pytest.skip("Reference seq_des CSV not found")

    df = pd.read_csv(csv_path)
    cal = make_caliby(0, True)

    grouped = df.groupby("example_id")

    # Only check the first example_id from the reference CSV
    # (Because random number behavior differs with batched design calls.)
    example_id, group = next(iter(grouped))
    content = read_pdb_content(example_id)
    if content is None:
        pytest.skip(f"Input file for {example_id} not found in tests/test_inputs/native_pdbs")

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


# unit design with position constraints
@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_design_with_pos_constraints():
    """Run design for the first example_id but apply position constraints from the CSV.
    This validates that `fixed_positions` are consumed by `Caliby.design`.
    """
    csv_path = Path("tests/expected_outputs/seq_des_output_pos_constraint.csv")
    constr_path = Path("tests/test_inputs/native_pdb_constraints.csv")

    if not csv_path.exists() or not constr_path.exists():
        pytest.skip("Reference constrained outputs or constraints CSV not found")

    df_out = pd.read_csv(csv_path)
    df_constr = pd.read_csv(constr_path)

    # Pick the first example_id from the constrained outputs
    first_example_id = df_out.loc[0, "example_id"]
    group = df_out[df_out["example_id"] == first_example_id]

    # Find matching constraint row (pdb_key matches example_id)
    pos_constraint_df = df_constr[df_constr["pdb_key"] == first_example_id]
    if pos_constraint_df.empty:
        pytest.skip(f"No constraints found for {first_example_id}")

    # Now run Caliby.design with these fixed positions and compare sequences
    cal = make_caliby(0, True)
    content = read_pdb_content(first_example_id)
    if content is None:
        pytest.skip(f"Input file for {first_example_id} not found in tests/test_inputs/native_pdbs")

    expected_rows = list(group.itertuples(index=False, name=None))
    num_expected = len(expected_rows)

    cal.set_seed(0, deterministic=True)
    results = cal.design(content, num_seqs=num_expected, pos_constraint_df=pos_constraint_df,)

    got_seqs = [r.get("seq", "") for r in results]
    expected_seqs = list(group["seq"])

    # Compare multisets (order may differ)
    assert sorted(got_seqs) == sorted(expected_seqs), (
        f"Constrained design sequences for {first_example_id} do not match expected outputs.\nGot: {got_seqs}\nExpected: {list(expected_seqs)}"
    )



@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing; skip integration tests")
def test_design_override_seq_positions_7xz3():
    """Test that per-position override (7xz3) forces residues 36-40 to 'C'."""
    constr_path = Path("tests/test_inputs/native_pdb_constraints.csv")
    if not constr_path.exists():
        pytest.skip("Constraints CSV not found")

    df_constr = pd.read_csv(constr_path)
    pos_constraint_df = df_constr[df_constr["pdb_key"] == "7xz3"]
    if pos_constraint_df.empty:
        pytest.skip("No constraint row for 7xz3")

    # load pdb
    data_dir = Path("tests/test_inputs/native_pdbs")
    p = None
    for ext in (".pdb", ".cif"):
        cand = data_dir / f"7xz3{ext}"
        if cand.exists():
            p = cand
            break
    if p is None:
        pytest.skip("Input PDB for 7xz3 not found")

    content = p.read_text()

    cal = make_caliby(0, True)
    results = cal.design(content, num_seqs=4, pos_constraint_df=pos_constraint_df)

    got_seqs = [r.get("seq", "") for r in results]

    for seq in got_seqs:
        assert seq[35:40] == "CCCCC", f"Override failed, got residues 36-40 as {seq[35:40]} instead of CCCCC"



#################
# Util functions#
#################

def parse_fixed_pos_seq(raw_fixed):
    """Parse strings like "A6-15,A20" into {'A': [6,7,...], ...}.

    Handles NaN/empty gracefully.
    """
    fixed_positions = {}
    if isinstance(raw_fixed, float) and np.isnan(raw_fixed):
        return fixed_positions

    raw = str(raw_fixed).strip()
    if not raw:
        return fixed_positions

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if len(part) < 2:
            continue
        chain = part[0]
        rest = part[1:]
        if "-" in rest:
            a, b = rest.split("-", 1)
            try:
                a_i = int(a)
                b_i = int(b)
            except Exception:
                continue
            rng = list(range(a_i, b_i + 1))
        else:
            try:
                rng = [int(rest)]
            except Exception:
                continue
        fixed_positions.setdefault(chain, []).extend(rng)

    return fixed_positions

def parse_override_seq(raw):
        out = {}
        if isinstance(raw, float) and np.isnan(raw):
            return out
        raw_s = str(raw).strip()
        if not raw_s:
            return out
        parts = [p.strip() for p in raw_s.split(",") if p.strip()]
        for part in parts:
            if ":" not in part or len(part) < 3:
                continue
            left, right = part.split(":", 1)
            chain = left[0]
            try:
                pos = int(left[1:])
            except Exception:
                continue
            out.setdefault(chain, {})[pos] = right.strip()
        return out