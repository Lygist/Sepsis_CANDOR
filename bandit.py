import argparse
import pickle
import torch
import numpy as np
import random
import os
from gym_sepsis.envs.SepsisWorld import SepsisWorld
from utils import generate_instances
from model import MLP
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.nn import MSELoss

def dm_plus_is(mlp, traj_ope, temp=5.0, device='cpu'):
    v_i_list = []
    real_rewards = []
    mlp.eval()
    with torch.no_grad():
        for traj in traj_ope:
            s = traj[0][0].float().to(device)
            a_i = traj[0][1].item()
            r_i = traj[0][2]
            probs_full = traj[0][4]
            real_rewards.append(r_i)
            inputs = torch.cat(
                [torch.cat((s.view(1, -1), one_hot(torch.tensor([a]), num_classes=25).float()), dim=1) for a in
                 range(25)], dim=0).to(device)
            hat_r = mlp(inputs).squeeze(1)
            # pi_e: softmax(hat_r / temp)
            pi_e = torch.softmax(hat_r / temp, dim=0)
            # DM term: sum_a pi_e(a) hat_r(a)
            dm = torch.sum(pi_e * hat_r)
            # IS correction
            pi_b_a_i = probs_full[0, a_i]
            pi_e_a_i = pi_e[a_i]
            rho = pi_e_a_i / pi_b_a_i if pi_b_a_i > 1e-6 else 0.0  # avoid div by zero
            hat_r_a_i = hat_r[a_i]
            correction = rho * (r_i - hat_r_a_i.item())
            # v_i
            v_i = dm + correction
            v_i_list.append(v_i.item())
    return v_i_list, real_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bandit Training Data for Sepsis and Evaluate DM+-IS')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--generate', type=int, default=0, help='Generate data or not. If 1 then invoke generate_instances function, if 0 then load data from before directly.')
    parser.add_argument('--sample-size', type=int, default=1, help='sample size')
    parser.add_argument('--number-trajectories', type=int, default=2000, help='number of trajectories')
    parser.add_argument('--cf-fraction', type=float, default=0.3, help='fraction of counterfactual samples')
    parser.add_argument('--cf-bias', type=float, default=1, help='bias for counterfactual rewards')
    parser.add_argument('--cf-noise-std', type=float, default=0.5, help='noise std for counterfactual rewards')
    parser.add_argument('--num-patients', type=int, default=1, help='number of patients')
    parser.add_argument('--effect-size', type=float, default=0.4, help='size of the action effect')
    parser.add_argument('--start-adhering', type=int, default=0, help='whether patients start by adhering or not')
    parser.add_argument('--discount', type=float, default=0.95, help='discount factor')
    parser.add_argument('--softness', type=float, default=5, help='softness of the DQN solver')
    parser.add_argument('--demonstrate-softness', type=float, default=0, help='demonstrate softness used to generate trajectories')
    parser.add_argument('--patient-data', type=str, default=None, help='location of the csv file containing patient data')
    args = parser.parse_args()
    print(args)

    seed = args.seed
    generate = args.generate
    sample_size = args.sample_size
    number_trajectories = args.number_trajectories
    cf_fraction = args.cf_fraction
    cf_bias = args.cf_bias
    cf_noise_std = args.cf_noise_std
    NUM_PATIENTS = args.num_patients
    EFFECT_SIZE = args.effect_size
    START_ADHERING = args.start_adhering
    discount = args.discount
    softness = args.softness
    demonstrate_softness = args.demonstrate_softness
    data_file = args.patient_data
    env_type = SepsisWorld

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_path = f'bandit_cf_dataset_seed{seed}_size{sample_size}_traj{number_trajectories}.pkl'
    if not generate and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            dataset, info = pickle.load(f)
        print(f"Loaded dataset from {data_path}")
    else:
        generate_kwargs = {
            'env_type': env_type,
            'NUM_PATIENTS': NUM_PATIENTS,
            'EFFECT_SIZE': EFFECT_SIZE,
            'discount': discount,
            'sample_size': sample_size,
            'num_trajectories': number_trajectories,
            'seed': seed,
            'softness': softness,
            'demonstrate_softness': demonstrate_softness,
            'START_ADHERING': START_ADHERING,
            'data_file': data_file,
            'cf_fraction': cf_fraction,
            'cf_bias': cf_bias,
            'cf_noise_std': cf_noise_std
        }
        dataset, info = generate_instances(**generate_kwargs)
        with open(data_path, 'wb') as f:
            pickle.dump((dataset, info), f)
        print(f"Dataset saved to {data_path}")

    # Split data: ensure pi_e is independent
    trajectories = dataset[0][4]
    traj_train = trajectories[:1000]
    traj_ope = trajectories[1000:]

    # Create D+ from traj_train
    D_plus = []
    for traj in traj_train:
        obs, action, reward, next_obs, probs_full, cf_data = traj[0]  # single step
        # Factual
        D_plus.append((obs, action, reward))
        # CF
        if cf_data is not None:
            for cf_action, cf_info in cf_data.items():
                if cf_info is not None:
                    cf_reward = cf_info['reward']
                    D_plus.append((obs, torch.tensor(cf_action), cf_reward))

    # Define MLP: input concat(s [1], onehot(a) [25]) -> [26], output scalar r
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channel_size_list = [1 + 25, 64, 1]  # state scalar + onehot(25)
    mlp = MLP(channel_size_list=channel_size_list).to(device)
    optimizer = Adam(mlp.parameters(), lr=1e-3)
    criterion = MSELoss()

    # Train MLP on D+
    epochs = 50
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
    ope_results, real_rewards = dm_plus_is(mlp, traj_ope, temp=1.0, device=device)

    # Save result
    result_data = {'ope': ope_results, 'real': real_rewards}
    result_path = f'cf_bandit_results_seed{seed}_traj{number_trajectories}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(result_data, f)
    print(f"OPE results saved to {result_path}")