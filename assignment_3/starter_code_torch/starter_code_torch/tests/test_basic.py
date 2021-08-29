import unittest
import code
from code.baseline_network import BaselineNetwork
from code.network_utils import build_mlp, np2torch
from code.policy_gradient import PolicyGradient
from code.config import get_config
import gym
import numpy as np
import torch
import builtins

import pytest

# Suppress unnecessary logging
# gym.logging.disable(gym.logging.FATAL)
builtins.config = None

class TestBasic(unittest.TestCase):
    @pytest.fixture
    def setUp(self):
        self.pg = None
        builtins.config = None
        
    @pytest.fixture
    def setUpEnv(self, env_name):
        config = get_config(env_name, True)
        self.env = gym.make(config.env_name)
        builtins.config = config
        self.pg = PolicyGradient(self.env, config, seed=1)
        self.policy = self.pg.policy
        self.baseline = BaselineNetwork(self.env, config)
        self.rand_obs = np.random.randn(10, self.pg.observation_dim)

    def test_policy_optimizer_exists(self):
        self.setUpEnv('cartpole')
        self.assertTrue(hasattr(self.pg, 'optimizer'))
        self.assertTrue(isinstance(self.pg.optimizer, torch.optim.Optimizer))

    def test_baseline_optimizer_exists(self):
        self.setUpEnv('cartpole')
        self.assertTrue(hasattr(self.baseline, 'optimizer'))
        self.assertTrue(isinstance(self.baseline.optimizer, torch.optim.Optimizer))

    def test_get_returns_zero(self):
        self.setUpEnv('cartpole')
        paths = [{'reward': np.zeros(11)}]
        returns = self.pg.get_returns(paths)
        expected = np.zeros(11)
        self.assertEqual(returns.shape, (11,))
        diff = np.sum((returns - expected)**2)
        self.assertAlmostEqual(diff, 0, delta=0.01)

    def test_get_returns_ones(self):
        self.setUpEnv('cartpole')
        paths = [{'reward': np.ones(5)}]
        returns = self.pg.get_returns(paths)
        gamma = self.pg.config.gamma
        expected = np.array([
            1+gamma+gamma**2+gamma**3+gamma**4,
            1+gamma+gamma**2+gamma**3,
            1+gamma+gamma**2,
            1+gamma,
            1
        ])
        diff = np.sum((returns - expected)**2)
        self.assertAlmostEqual(diff, 0, delta=0.001)

    def _test_sampled_actions(self):
        actions = self.policy.act(self.rand_obs)
        action_space = self.env.action_space
        discrete = isinstance(action_space, gym.spaces.Discrete)
        for action in actions:
            if discrete:
                self.assertTrue(action_space.contains(action))
            else:
                # We don't use contains because technically the Gaussian policy
                # doesn't respect the action bounds
                self.assertEqual(action_space.shape, action.shape)

    def test_cartpole_sampled_actions(self):
        self.setUpEnv('cartpole')
        self._test_sampled_actions()

    def test_pendulum_sampled_actions(self):
        self.setUpEnv('pendulum')
        self._test_sampled_actions()

    def test_cheetah_sampled_actions(self):
        self.setUpEnv('cheetah')
        self._test_sampled_actions()

    def _test_log_prob(self):
        actions = np2torch(self.policy.act(self.rand_obs))
        observations = np2torch(self.rand_obs)
        log_probs = self.policy.action_distribution(observations).log_prob(actions).detach()
        self.assertEqual(log_probs.shape, torch.Size([len(observations)]))

    def test_cartpole_logprob(self):
        self.setUpEnv('cartpole')
        self._test_log_prob()

    def test_pendulum_logprob(self):
        self.setUpEnv('pendulum')
        self._test_log_prob()

    def test_cheetah_logprob(self):
        self.setUpEnv('cheetah')
        self._test_log_prob()

    def test_policy_network_cartpole_logprob_value(self):
        self.setUpEnv('cartpole')
        actions = np2torch(self.policy.act(self.rand_obs))
        observations = np2torch(self.rand_obs)
        log_probs = self.policy.action_distribution(observations).log_prob(actions).detach()
        self.assertTrue(torch.all(log_probs < 0))

    def test_baseline_op(self):
        self.setUpEnv('cartpole')
        # make sure we can overfit!
        returns = np.random.randn(len(self.rand_obs))
        for i in range(1000):
            self.baseline.update_baseline(returns, self.rand_obs)
        advantages = self.baseline.calculate_advantage(returns, self.rand_obs)
        self.assertTrue(np.allclose(advantages, 0, atol=0.01))

    def test_adv_basic(self):
        self.setUpEnv('cartpole')
        returns = np.random.randn(5)
        observations = np.random.randn(5,4)
        self.pg.config.use_baseline = False
        self.pg.config.normalize_advantage = False
        res = self.pg.calculate_advantage(returns, observations)
        self.assertAlmostEqual(np.sum(res), np.sum(returns), delta=0.001)