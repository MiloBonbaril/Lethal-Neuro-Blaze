
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys
import os
import shutil

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import agent
import config

class TestReplayBuffer(unittest.TestCase):
    def test_push_and_sample(self):
        capacity = 10
        buffer = agent.ReplayBuffer(capacity)
        
        state = np.zeros((4, 84, 84))
        action = 1
        reward = 1.0
        next_state = np.zeros((4, 84, 84))
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        self.assertEqual(len(buffer), 1)
        
        batch_size = 1
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        self.assertEqual(states.shape, (1, 4, 84, 84))
        self.assertEqual(actions[0], 1)
        self.assertEqual(rewards[0], 1.0)
        
    def test_capacity(self):
        capacity = 2
        buffer = agent.ReplayBuffer(capacity)
        
        buffer.push(np.zeros((4,84,84)), 0, 0, np.zeros((4,84,84)), False)
        buffer.push(np.zeros((4,84,84)), 1, 0, np.zeros((4,84,84)), False)
        buffer.push(np.zeros((4,84,84)), 2, 0, np.zeros((4,84,84)), False)
        
        self.assertEqual(len(buffer), 2)
        # Should have discarded the first one, let's just check size for now

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.input_shape = (4, 84, 84)
        self.num_actions = 7
        # Mocking Brain within Agent might be hard without dependency injection.
        # However, we can patch 'agent.Brain' class if needed, but let's use real Brain for integration testing
        # as it is small.
        # We need to mock torch.device to ensure CPU usage for tests
        self.patcher = patch('torch.cuda.is_available', return_value=False)
        self.mock_cuda = self.patcher.start()
        
    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        my_agent = agent.Agent(self.input_shape, self.num_actions)
        self.assertIsNotNone(my_agent.policy_net)
        self.assertIsNotNone(my_agent.target_net)
        self.assertIsNotNone(my_agent.optimizer)
        self.assertIsNotNone(my_agent.memory)

    @patch('random.random')
    def test_select_action_exploitation(self, mock_random):
        my_agent = agent.Agent(self.input_shape, self.num_actions)
        my_agent.epsilon = 0.1
        mock_random.return_value = 0.5 # > epsilon => Exploitation
        
        state = np.zeros(self.input_shape)
        action = my_agent.select_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.num_actions)

    @patch('random.random')
    def test_select_action_exploration(self, mock_random):
        my_agent = agent.Agent(self.input_shape, self.num_actions)
        my_agent.epsilon = 0.9
        mock_random.return_value = 0.1 # < epsilon => Exploration
        
        state = np.zeros(self.input_shape)
        action = my_agent.select_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.num_actions)

    @patch('torch.save')
    def test_save(self, mock_save):
        my_agent = agent.Agent(self.input_shape, self.num_actions)
        filename = "test_checkpoint.pth"
        episode = 10
        best_reward = 100
        
        my_agent.save(filename, episode, best_reward)
        
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        checkpoint = args[0]
        self.assertEqual(checkpoint['episode'], episode)
        self.assertEqual(checkpoint['best_reward'], best_reward)

    @patch('torch.load')
    @patch('os.path.exists', return_value=True)
    def test_load(self, mock_exists, mock_load):
        my_agent = agent.Agent(self.input_shape, self.num_actions)
        filename = "test_checkpoint.pth"
        
        # Mock a loaded checkpoint
        mock_checkpoint = {
            'model_state': my_agent.policy_net.state_dict(),
            'optimizer_state': my_agent.optimizer.state_dict(),
            'epsilon': 0.5,
            'episode': 20,
            'best_reward': 200
        }
        mock_load.return_value = mock_checkpoint
        
        start_episode, best_reward = my_agent.load(filename)
        
        self.assertEqual(start_episode, 21) # episode + 1
        self.assertEqual(best_reward, 200)
        self.assertEqual(my_agent.epsilon, 0.5)

    def test_learn_step(self):
        """Test one step of learning."""
        my_agent = agent.Agent(self.input_shape, self.num_actions)
        
        # We need enough samples in memory
        for _ in range(config.BATCH_SIZE + 1):
            state = np.zeros(self.input_shape)
            next_state = np.zeros(self.input_shape)
            my_agent.memory.push(state, 1, 1.0, next_state, False)
            
        loss = my_agent.learn()
        self.assertIsInstance(loss, float)
        
if __name__ == '__main__':
    unittest.main()
