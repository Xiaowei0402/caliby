import sys
import unittest
from unittest.mock import MagicMock
import shutil
from pathlib import Path
import pandas as pd
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
import torch
import os

# Add current directory to path so we can import caliby
# Assuming this file is in tests/ folder, and caliby module is in parent folder
sys.path.append(str(Path(__file__).parent.parent))

from caliby.eval.eval_utils.seq_des_utils import score_samples, run_seq_des, get_seq_des_model
from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files

class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).parent.parent
        cls.examples_dir = cls.base_dir / "examples"
        cls.output_dir = cls.base_dir / "tests_output"
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.output_dir.mkdir()
        
        # Ensure we are at base dir for relative paths in config to work
        os.chdir(cls.base_dir)

    def tearDown(self):
        GlobalHydra.instance().clear()

    def get_config(self, config_name, overrides):
        # Path to the config directory relative to this file
        config_path = self.base_dir / "caliby/configs/eval/sampling"
        # We use relative path for config_dir since we chdir-ed to base_dir, NO, initialize_config_dir needs absolute path usually
        abs_config_path = config_path.resolve()
        
        GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(config_dir=str(abs_config_path), version_base="1.3.2"):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
        return cfg

    def test_01_score(self):
        print("Testing score example...")
        out_dir = self.output_dir / "score"
        overrides = [
            "ckpt_path=model_params/caliby/caliby.ckpt",
            f"input_cfg.pdb_dir={self.examples_dir}/example_data/native_pdbs",
            f"out_dir={out_dir}"
        ]
        
        cfg = self.get_config("score", overrides)
        
        # 1. Get PDB files
        pdb_files = get_pdb_files(**cfg.input_cfg)
        self.assertTrue(len(pdb_files) > 0, "No PDB files found")
        
        # 2. Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
        
        # 4. Run score_samples
        
        outputs = score_samples(
            model=seq_des_model["model"],
            data_cfg=seq_des_model["data_cfg"],
            sampling_cfg=seq_des_model["sampling_cfg"],
            pdb_paths=pdb_files,
            device=device,
        )
        
        # 5. Verify outputs
        self.assertIn("example_id", outputs)
        self.assertIn("seq", outputs)
        self.assertIn("U", outputs)
        self.assertEqual(len(outputs["example_id"]), len(pdb_files))
        
        # Save to csv (mimic script)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame({"example_id": outputs["example_id"], "seq": outputs["seq"], "U": outputs["U"]})
        output_df.to_csv(out_dir / "score_outputs.csv", index=False)
        print(f"Score outputs saved to {out_dir / 'score_outputs.csv'}")

    def test_02_seq_des_multi(self):
        print("Testing seq_des_multi example...")
        out_dir = self.output_dir / "seq_des_multi"
        overrides = [
            "ckpt_path=model_params/caliby/caliby.ckpt",
            f"input_cfg.pdb_dir={self.examples_dir}/example_data/native_pdbs",
            "sampling_cfg_overrides.num_seqs_per_pdb=4",
            "sampling_cfg_overrides.batch_size=2", # Reduce batch size for test
            "sampling_cfg_overrides.num_workers=1", # Avoid parallel issues in test
            f"out_dir={out_dir}"
        ]
        
        cfg = self.get_config("seq_des_multi", overrides)
        
        pdb_files = get_pdb_files(**cfg.input_cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
        
        pos_constraint_df = None
        
        outputs = run_seq_des(
            model=seq_des_model["model"],
            data_cfg=seq_des_model["data_cfg"],
            sampling_cfg=seq_des_model["sampling_cfg"],
            pdb_paths=pdb_files,
            device=device,
            pos_constraint_df=pos_constraint_df,
            out_dir=str(out_dir),
        )
        
        # Verify
        self.assertIn("out_pdb", outputs)
        self.assertIn("seq", outputs)
        # Expected count: num_pdbs * num_seqs_per_pdb
        expected_count = len(pdb_files) * 4
        self.assertEqual(len(outputs["seq"]), expected_count)

    def test_03_seq_des_multi_constraints(self):
        print("Testing seq_des_multi_constraints example...")
        out_dir = self.output_dir / "seq_des_multi_constraints"
        constraint_csv = self.examples_dir / "example_data/pos_constraint_csvs/native_pdb_constraints.csv"
        
        overrides = [
            "ckpt_path=model_params/caliby/caliby.ckpt",
            f"input_cfg.pdb_dir={self.examples_dir}/example_data/native_pdbs",
            f"pos_constraint_csv={constraint_csv}",
            "sampling_cfg_overrides.num_seqs_per_pdb=2",
            "sampling_cfg_overrides.num_workers=1",
            f"out_dir={out_dir}"
        ]
        
        cfg = self.get_config("seq_des_multi", overrides)
        
        pdb_files = get_pdb_files(**cfg.input_cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)
        
        # Load constraints
        if cfg.pos_constraint_csv is not None:
            pos_constraint_df = pd.read_csv(cfg.pos_constraint_csv)
        else:
            pos_constraint_df = None
            
        self.assertIsNotNone(pos_constraint_df)
        
        outputs = run_seq_des(
            model=seq_des_model["model"],
            data_cfg=seq_des_model["data_cfg"],
            sampling_cfg=seq_des_model["sampling_cfg"],
            pdb_paths=pdb_files,
            device=device,
            pos_constraint_df=pos_constraint_df,
            out_dir=str(out_dir),
        )
        
        self.assertTrue(len(outputs["seq"]) > 0)

if __name__ == '__main__':
    unittest.main()
