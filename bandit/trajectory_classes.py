import numpy as np

'''
DESCRIPTION: Defines data structures for handlig trajectories and annotations. 
'''

class AnnotatedTrajectory:
    pass  # TODO


class Trajectory:
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        num_possible_states: int = 2,
        num_possible_actions: int = 2,
    ):
        self.trajectory_len = len(states)
        assert len(actions) == self.trajectory_len
        assert len(rewards) == self.trajectory_len

        self.states = states
        self.actions = actions
        self.rewards = rewards

        self.num_possible_states = num_possible_states
        self.num_possible_actions = num_possible_actions

    def create_nan_expanded_rewards(self):
        """
        Expands the rewards from shape (trajectory_len) to shape (num_actions,
        trajectory_len), where actions that were not taken have value nan.
        """
        output = np.full((self.trajectory_len, self.num_possible_actions), np.nan)
        # There's room for optimization down below, but it will save little time.
        for timestep, _, action, reward in self:
            output[timestep, action] = reward
        return output

    def unpack(self):
        """
        Convenience function for retrieving the states, actions, and rewards as
        separate arrays.
        """
        return self.states, self.actions, self.rewards

    def __iter__(self):
        """
        Allows for iteration through the trajectory's timesteps, retrieving
        (timestep, state, action, reward) tuples for each timestep.
        """
        for i in range(self.trajectory_len):
            yield i, self.states[i], self.actions[i], self.rewards[i]

    def __len__(self):
        return self.trajectory_len
