from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from gym_sepsis.envs.sepsis_env import SepsisEnv

env = SepsisEnv()
model = DQN.load("dqn_baseline")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     next_obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()