
import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brain
import config

class TestBrain(unittest.TestCase):
    def setUp(self):
        self.input_shape = (4, 84, 84)
        self.num_actions = 7

    def test_brain_initialization(self):
        """Test if Brain initializes with correct layers."""
        model = brain.Brain(self.input_shape, self.num_actions)
        
        # Check layers existence
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'conv2'))
        self.assertTrue(hasattr(model, 'conv3'))
        self.assertTrue(hasattr(model, 'fc1'))
        self.assertTrue(hasattr(model, 'fc2'))
        
        # Check output size of final layer
        self.assertEqual(model.fc2.out_features, self.num_actions)

    def test_brain_forward(self):
        """Test forward pass output shape."""
        model = brain.Brain(self.input_shape, self.num_actions)
        batch_size = 2
        dummy_input = torch.zeros(batch_size, *self.input_shape)
        
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, self.num_actions))

    def test_device_compatibility(self):
        """Test if model moves to CPU/GPU correctly."""
        device = torch.device("cpu") # Force CPU for this test to be safe
        model = brain.Brain(self.input_shape, self.num_actions).to(device)
        dummy_input = torch.zeros(1, *self.input_shape).to(device)
        output = model(dummy_input)
        self.assertEqual(output.device.type, 'cpu')

class TestMotorCortex(unittest.TestCase):
    @patch('brain.pydirectinput')
    def test_initialization_disables_failsafe(self, mock_pdi):
        """Test if MotorCortex disables failsafe on init."""
        cortex = brain.MotorCortex()
        self.assertFalse(mock_pdi.FAILSAFE)

    @patch('brain.pydirectinput')
    def test_execute_presses_key(self, mock_pdi):
        """Test if execute maps index to key press."""
        cortex = brain.MotorCortex()
        
        # Action 3 is 'space' in config.ACTION_MAP
        # Make sure config.ACTION_MAP[3] is indeed 'space' or whatever is in config
        action_idx = 3
        expected_key = config.ACTION_MAP.get(action_idx)
        
        cortex.execute(action_idx)
        
        if expected_key:
            mock_pdi.press.assert_called_with(expected_key)
        else:
             mock_pdi.press.assert_not_called()

    @patch('brain.pydirectinput')
    def test_execute_noop(self, mock_pdi):
        """Test if execute does nothing for None action."""
        cortex = brain.MotorCortex()
        
        # Action 0 is usually No-Op (None)
        action_idx = 0
        expected_key = config.ACTION_MAP.get(action_idx)
        
        if expected_key is None:
            cortex.execute(action_idx)
            mock_pdi.press.assert_not_called()

if __name__ == '__main__':
    unittest.main()
