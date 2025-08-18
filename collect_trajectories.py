import numpy as np
import pickle
import random
from tqdm import tqdm
from gym_sepsis.envs.SepsisEnv import SepsisEnv

NUM_TRAJECTORIES = 20

env = SepsisEnv()

original_trajectories = []
pbar = tqdm(total=NUM_TRAJECTORIES, desc="Original Trajectories")
while len(original_trajectories) < NUM_TRAJECTORIES:
    obs = env.reset()
    trajectory = []
    while True:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        trajectory.append((obs, action, reward, next_obs))
        obs = next_obs
        if done:
            break
    original_trajectories.append(trajectory)
    pbar.update(1)
# with open("original_trajectories.pkl", "wb") as f:
#     pickle.dump(original_trajectories, f)
# print(f"Number of original trajectories: {len(original_trajectories)}")
# print(f"Average length of trajectory: {sum([len(trajectory) for trajectory in original_trajectories]) / len(original_trajectories)}")

trajectories = random.sample(original_trajectories, int(NUM_TRAJECTORIES * 0.1))
perfect_counterfactual_trajectories = []
pbar = tqdm(total=int(NUM_TRAJECTORIES * 0.1), desc="Perfect Counterfactuals")
for original_trajectory in trajectories:
    obs = env.reset(original_trajectory[0][0])
    trajectory = []
    action = np.random.choice([a for a in range(env.action_space.n) if a != original_trajectory[0][1]])
    next_obs, reward, done, info = env.step(action)
    trajectory.append((obs, action, reward, next_obs))
    obs = next_obs
    if not done:
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs, action, reward, next_obs))
            obs = next_obs
            if done:
                break
    perfect_counterfactual_trajectories.append(trajectory)
    pbar.update(1)
# with open("perfect_counterfactual_trajectories.pkl", "wb") as f:
#     pickle.dump(perfect_counterfactual_trajectories, f)
# print(f"Number of counterfactual trajectories: {len(perfect_counterfactual_trajectories)}")
# print(f"Average length of counterfactual trajectory: {sum([len(trajectory) for trajectory in perfect_counterfactual_trajectories]) / len(perfect_counterfactual_trajectories)}")

trajectories = random.sample(original_trajectories, int(NUM_TRAJECTORIES * 0.1))
imperfect_counterfactual_trajectories = []
pbar = tqdm(total=int(NUM_TRAJECTORIES * 0.1), desc="Imperfect Counterfactuals")
for original_trajectory in trajectories:
    obs = env.reset(original_trajectory[0][0])
    trajectory = []
    action = np.random.choice([a for a in range(env.action_space.n) if a != original_trajectory[0][1]])
    next_obs, reward, done, info = env.step(action)
    trajectory.append((obs, action, reward, next_obs))
    obs = next_obs
    if not done:
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            reward = np.random.normal(reward + 2, 1)
            trajectory.append((obs, action, reward, next_obs))
            obs = next_obs
            if done:
                break
    imperfect_counterfactual_trajectories.append(trajectory)
    pbar.update(1)
# with open("imperfect_counterfactual_trajectories.pkl", "wb") as f:
#     pickle.dump(imperfect_counterfactual_trajectories, f)
# print(f"Number of counterfactual trajectories: {len(imperfect_counterfactual_trajectories)}")
# print(f"Average length of counterfactual trajectory: {sum([len(trajectory) for trajectory in imperfect_counterfactual_trajectories]) / len(imperfect_counterfactual_trajectories)}")