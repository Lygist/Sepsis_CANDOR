import pickle
import torch
import numpy as np
import random
import os
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.nn import MSELoss
from model import MLP

def dm_plus_is(mlp, ope_steps, temp=1.0, device='cpu'):
    v_i_list = []
    real_rewards = []
    mlp.eval()
    with torch.no_grad():
        for step in ope_steps:
            s, a_i_tensor, r_i, _, probs_full, cf_data = step
            a_i = a_i_tensor.item()
            real_rewards.append(r_i)
            s = s.float().to(device)
            inputs = torch.cat([torch.cat((s.view(1, -1), one_hot(torch.tensor([a]), num_classes=25).float()), dim=1) for a in range(25)], dim=0).to(device)
            hat_r = mlp(inputs).squeeze(1)
            pi_e = torch.softmax(hat_r / temp, dim=0)
            # DM term: sum_a pi_e(a) hat_r(a)
            dm = torch.sum(pi_e * hat_r)
            # IS correction
            pi_b_a_i = probs_full[0, a_i]
            pi_e_a_i = pi_e[a_i]
            rho = pi_e_a_i / pi_b_a_i if pi_b_a_i > 1e-6 else 0.0
            hat_r_a_i = hat_r[a_i]
            correction = rho * (r_i - hat_r_a_i.item())

            v_i = dm + correction
            v_i_list.append(v_i.item())
    return v_i_list, real_rewards


if __name__ == "__main__":
    data_path = 'cf_dataset_seed1_size10.pkl'
    with open(data_path, 'rb') as f:
        dataset, info = pickle.load(f)
    print(f"Loaded dataset from {data_path}")

    # Mix all samples' trajectories
    all_trajectories = [traj for sample in dataset for traj in sample[4]]

    # Flatten all steps
    flat_steps = [step for traj in all_trajectories for step in traj]

    # Split: 80% train, 20% ope
    N = len(flat_steps)
    train_size = int(0.8 * N)
    train_steps = flat_steps[:train_size]
    ope_steps = flat_steps[train_size:]

    # Create D+ from train_steps
    D_plus = []
    for step in train_steps:
        obs, action, reward, next_obs, probs_full, cf_data = step
        D_plus.append((obs, action, reward))
        if cf_data is not None:
            for cf_action, cf_info in cf_data.items():
                if cf_info is not None:
                    cf_reward = cf_info['reward']
                    D_plus.append((obs, torch.tensor(cf_action), cf_reward))

    # Define MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channel_size_list = [1 + 25, 64, 1]
    mlp = MLP(channel_size_list=channel_size_list).to(device)
    optimizer = Adam(mlp.parameters(), lr=1e-3)
    criterion = MSELoss()

    # Train MLP on D+
    epochs = 20
    mlp.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for obs, action, reward in D_plus:
            optimizer.zero_grad()
            input_tensor = torch.cat((obs.float().view(1, -1), one_hot(action.view(1, -1), num_classes=25).float().view(1, -1)), dim=1).to(device)
            pred = mlp(input_tensor).squeeze()
            loss = criterion(pred, torch.tensor([reward]).float().to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(D_plus)}")

    # Evaluate DM+-IS
    ope_results, real_rewards = dm_plus_is(mlp, ope_steps, temp=1.0, device=device)

    # Save results
    result_data = {'ope': ope_results, 'real': real_rewards}
    result_path = 'bandit_results_seed1_size10_flattened.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(result_data, f)
    print(f"OPE results and real rewards saved to {result_path}")