import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import importlib.util
import sys

try:
    import torch
except Exception:
    pytest.skip("torch missing; skip integration tests", allow_module_level=True)

repo_root = Path(__file__).resolve().parents[1]
caliby_file = repo_root / "caliby/Caliby.py"
spec = importlib.util.spec_from_file_location("caliby_local", str(caliby_file))
caliby_mod = importlib.util.module_from_spec(spec)
sys.modules["caliby_local"] = caliby_mod
spec.loader.exec_module(caliby_mod)
Caliby = caliby_mod.Caliby

CKPT = Path("model_params/caliby/caliby.ckpt")

def make_caliby(seed: int = 42, deterministic: bool = True):
    return Caliby(str(CKPT), seed=seed, deterministic=deterministic)

def read_pdb_content():
    # Use a known PDB from tests
    data_dir = Path("tests/test_inputs/native_pdbs")
    # Just grab any PDB/CIF file that exists
    for f in data_dir.glob("7xhz.*"):
        if f.suffix in (".pdb", ".cif"):
            return f.read_text()
    return None

def calculate_diversity(sequences):
    """Calculate average fractional Hamming distance between unique pairs of sequences."""
    if len(sequences) < 2:
        return 0.0
    
    unique_seqs = list(set(sequences))
    if len(unique_seqs) < 2:
        return 0.0
        
    distances = []
    for i in range(len(unique_seqs)):
        for j in range(i + 1, len(unique_seqs)):
            s1, s2 = unique_seqs[i], unique_seqs[j]
            # Assume same length for simplicity if they come from same structure
            dist = sum(1 for a, b in zip(s1, s2) if a != b) / max(len(s1), len(s2))
            distances.append(dist)
    return np.mean(distances)

@pytest.mark.skipif(not (repo_root / CKPT).exists(), reason=f"checkpoint missing at {repo_root / CKPT}")
def test_temperature_diversity():
    cal = Caliby(str(repo_root / CKPT), seed=42, deterministic=True)
    content = read_pdb_content()
    if content is None:
        pytest.skip("No input structure found for testing")

    num_seqs = 5
    
    # Test temperatures
    temp_low = 0.01
    temp_mid = 0.1
    temp_high = 0.5
    
    # Generate for low temperature
    results_low = cal.design(content, num_seqs=num_seqs, temperature=temp_low)
    seqs_low = [r["seq"] for r in results_low]
    div_low = calculate_diversity(seqs_low)
    
    # Generate for mid temperature
    results_mid = cal.design(content, num_seqs=num_seqs, temperature=temp_mid)
    seqs_mid = [r["seq"] for r in results_mid]
    div_mid = calculate_diversity(seqs_mid)
    
    # Generate for high temperature
    results_high = cal.design(content, num_seqs=num_seqs, temperature=temp_high)
    seqs_high = [r["seq"] for r in results_high]
    div_high = calculate_diversity(seqs_high)
    
    print(f"\nDiversity at T={temp_low}: {div_low:.4f}")
    print(f"Diversity at T={temp_mid}: {div_mid:.4f}")
    print(f"Diversity at T={temp_high}: {div_high:.4f}")
    
    # We expect diversity to generally increase with temperature
    # Note: with small sample sizes (num_seqs=5), stochastic noise might cause small fluctuations.
    # We primarily want to see that T=0.5 is significantly more diverse than T=0.01.
    assert div_high > div_low, f"High temperature {temp_high} should have higher diversity than {temp_low}"
    
    # Optional: check unique sequence count
    unique_low = len(set(seqs_low))
    unique_high = len(set(seqs_high))
    assert unique_high >= unique_low, "Higher temperature should yield at least as many unique sequences as lower temperature"
