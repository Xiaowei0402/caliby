import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import torch
from omegaconf import OmegaConf

from caliby.inference import Caliby

class TestCalibyInference(unittest.TestCase):
    @patch('caliby.inference.LitSeqDenoiser')
    @patch('caliby.inference.get_cfg_from_ckpt')
    @patch('caliby.inference.hydra.utils.instantiate')
    @patch('caliby.inference.aw_parse')
    def test_initialization_and_design(self, mock_aw_parse, mock_instantiate, mock_get_cfg, mock_lit_model):
        
        # Mock dependencies
        mock_lit_instance = MagicMock()
        # Important: chained call .eval() must return the instance itself
        mock_lit_instance.eval.return_value = mock_lit_instance 
        
        mock_lit_model.load_from_checkpoint.return_value = mock_lit_instance
        
        mock_lit_instance.model = MagicMock()
        # Configure sample return
        mock_lit_instance.model.sample.return_value = (
            {"input_pdb": [MagicMock()]}, # atom_arrays
            {"input_pdb": [{"U": 1.0}]}   # aux
        )
        
        mock_get_cfg.return_value = (OmegaConf.create({"data": {}}), None)
        mock_instantiate.return_value = OmegaConf.create({})
        
        # Mock aw_parse return structure
        mock_aw_parse.return_value = {
            "assemblies": {"1": [MagicMock()]},
            "chain_info": {"A": {"processed_entity_canonical_sequence": "ACDEF"}}
        }
        
        # Initialize Caliby
        model = Caliby("dummy.ckpt", device="cpu")
        
        # Test loading
        mock_lit_model.load_from_checkpoint.assert_called()
        
        # Test design
        pdb_content = "ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N"
        
        # We need to mock preprocess_transform and sd_featurizer as they complex
        with patch('caliby.inference.preprocess_transform') as mock_prep, \
             patch('caliby.inference.sd_featurizer') as mock_feat, \
             patch('caliby.inference.to_cif_string') as mock_to_cif, \
             patch('caliby.inference.non_rcsb.initialize_chain_info_from_atom_array') as mock_chain_info:
            
            mock_prep.return_value = MagicMock() # Pipeline
            mock_prep.return_value.return_value = {} # Pipeline output
            
            mock_feat.return_value = MagicMock() # Featurizer
            # Ensure boolean masks are used where appropriate
            mock_feat.return_value.return_value = {
                "is_protein": torch.tensor([True], dtype=torch.bool), 
                "is_atomized": torch.tensor([False], dtype=torch.bool),
                "token_pad_mask": torch.tensor([1]),
                "token_resolved_mask": torch.tensor([0]), # dummy
                "prot_bb_atom_mask": torch.tensor([0]),
                "atom_to_token_map": torch.tensor([0]),
                "atom_pad_mask": torch.tensor([1]),
                "atom_resolved_mask": torch.tensor([0]),
                "example_id": "input_pdb"
            } # Featurizer output (dummy dict for masking)
            
            mock_to_cif.return_value = "dummy_cif"
            mock_chain_info.return_value = {"A": {"processed_entity_canonical_sequence": "ACDEF"}}

            # Run design
            results = model.design(pdb_content, num_seqs=1)
            
            # Assertions
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['seq'], "ACDEF")
            self.assertEqual(results[0]['pdb_string'], "dummy_cif")
            self.assertEqual(results[0]['scores']['U'], 1.0)
            
            # Check if aw_parse is called with a temp file
            # mock_aw_parse (path, ...)
            self.assertTrue(mock_aw_parse.called)
            
if __name__ == '__main__':
    unittest.main()
