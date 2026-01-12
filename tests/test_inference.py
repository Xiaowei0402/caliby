import unittest
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Caliby import Caliby

class TestCalibyInferenceReal(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).parent.parent
        self.ckpt_path = self.base_dir / "model_params/caliby/caliby.ckpt"
        self.pdb_path = self.base_dir / "examples/example_data/native_pdbs/7xhz.cif"
        
        if not self.ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {self.ckpt_path}")
        if not self.pdb_path.exists():
            self.skipTest(f"PDB file not found at {self.pdb_path}")

    def test_design_real(self):
        print("Initializing Caliby with real weights...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Override sampling config for speed
        sampling_overrides = {
            "num_seqs_per_pdb": 2,
            "batch_size": 1,
            "num_workers": 0
        }
        
        model = Caliby(str(self.ckpt_path), device=device, sampling_overrides=sampling_overrides)
        
        with open(self.pdb_path, 'r') as f:
            pdb_content = f.read()
            
        print(f" Designing sequence for {self.pdb_path.name}...")
        # num_seqs in design method overrides sampling_overrides['num_seqs_per_pdb'] if implemented correctly,
        # otherwise sampling_overrides takes precedence. Let's check Caliby.design signature.
        # It takes num_seqs argument.
        results = model.design(pdb_content, num_seqs=2)
        
        self.assertEqual(len(results), 2)
        for i, res in enumerate(results):
            print(f" Sample {i+1}:")
            print(f"  Sequence length: {len(res['seq'])}")
            print(f"  Score: {res['scores']['U']}")
            
            self.assertIn('seq', res)
            self.assertIn('pdb_string', res)
            self.assertIn('scores', res)
            self.assertTrue(len(res['seq']) > 0)

if __name__ == '__main__':
    unittest.main()
