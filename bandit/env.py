import numpy as np
import gym
from gym import spaces


class TabularMDPEnv(gym.Env):
    """
    A custom Gymnasium environment for a tabular MDP.
    Fixed Horizon.
    """

    def __init__(
            self,
            num_states: int,
            num_actions: int,
            horizon: int,
            transition_probs: np.ndarray,
            reward_means: np.ndarray,
            reward_stds: np.ndarray,
            initial_state_dist: np.ndarray,
            seed: int = None
    ):
        super(TabularMDPEnv, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.horizon = horizon

        # Define spaces
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        # Dynamics
        # Shape: (S, A, S')
        self.transition_probs = transition_probs
        # Shape: (S, A)
        self.reward_means = reward_means
        self.reward_stds = reward_stds
        self.initial_state_dist = initial_state_dist

        self._current_step = 0
        self._current_state = 0
        self.seed(seed)
        self._rng = np.random.default_rng(seed)

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._current_state = self._rng.choice(
            self.num_states, p=self.initial_state_dist
        )
        return self._current_state

    def step(self, action):
        # 1. Sample Reward
        mean = self.reward_means[self._current_state, action]
        std = self.reward_stds[self._current_state, action]
        reward = self._rng.normal(mean, std)

        # 2. Transition
        next_state = self._rng.choice(
            self.num_states,
            p=self.transition_probs[self._current_state, action]
        )

        # 3. Update step and state
        self._current_step += 1
        self._current_state = next_state

        # 4. Check termination
        done = False
        if self._current_step >= self.horizon:
            done = True

        return self._current_state, float(reward), done, {}

    def set_biased_rewards(self, bias_mean, bias_std_add):
        """Helper to inject bias for 'Imperfect' annotator training."""
        self.reward_means = self.reward_means + bias_mean
        self.reward_stds = self.reward_stds + bias_std_add