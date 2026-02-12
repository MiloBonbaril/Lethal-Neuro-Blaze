
import unittest
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

class TestConfig(unittest.TestCase):
    def test_game_settings(self):
        """Test critical game settings."""
        self.assertIsInstance(config.WINDOW_TITLE, str)
        self.assertIsInstance(config.MENU_KEY, str)

    def test_hyperparameters(self):
        """Test hyperparameters ranges and types."""
        self.assertIsInstance(config.EPISODES, int)
        self.assertGreater(config.EPISODES, 0)
        self.assertIsInstance(config.MAX_STEPS_PER_EPISODE, int)
        self.assertGreater(config.MAX_STEPS_PER_EPISODE, 0)
        self.assertIsInstance(config.FRAME_SKIP, int)
        self.assertGreaterEqual(config.FRAME_SKIP, 1)

    def test_agent_hyperparameters(self):
        """Test agent specific hyperparameters."""
        self.assertIsInstance(config.BATCH_SIZE, int)
        self.assertGreater(config.BATCH_SIZE, 0)
        self.assertIsInstance(config.GAMMA, float)
        self.assertTrue(0 <= config.GAMMA <= 1)
        self.assertIsInstance(config.EPSILON_START, float)
        self.assertTrue(0 <= config.EPSILON_START <= 1)
        self.assertIsInstance(config.EPSILON_END, float)
        self.assertTrue(0 <= config.EPSILON_END <= 1)
        self.assertLess(config.EPSILON_END, config.EPSILON_START)

    def test_model_settings(self):
        """Test model configuration."""
        self.assertIsInstance(config.INPUT_SHAPE, tuple)
        self.assertEqual(len(config.INPUT_SHAPE), 3)
        self.assertEqual(config.INPUT_SHAPE[0], 4) # Channels

    def test_rewards(self):
        """Test reward values."""
        self.assertIsInstance(config.REWARD_SURVIVAL, float)
        self.assertIsInstance(config.REWARD_DAMAGE, float)
        self.assertIsInstance(config.REWARD_DEATH, float)
        self.assertIsInstance(config.REWARD_WIN, float)
        
        # Logic check: Winning should be positive, Death negative
        self.assertGreater(config.REWARD_WIN, 0)
        self.assertLess(config.REWARD_DEATH, 0)

    def test_action_map(self):
        """Test action mapping integrity."""
        self.assertIsInstance(config.ACTION_MAP, dict)
        # Ensure keys are continuous integers starting from 0
        keys = sorted(config.ACTION_MAP.keys())
        self.assertEqual(keys[0], 0)
        self.assertEqual(keys[-1], len(keys) - 1)
        self.assertEqual(len(keys), keys[-1] + 1)

if __name__ == '__main__':
    unittest.main()
