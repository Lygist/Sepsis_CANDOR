import gym
from stable_baselines3 import DQN
from gym_sepsis.envs.sepsis_env import SepsisEnv

env = SepsisEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
model.save("dqn_baseline")