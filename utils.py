import copy
import pickle

import tensorflow as tf
import numpy as np
import torch
import tqdm

from model import MLP
import random

# import sys
# sys.path.insert(1, '../')

from stable_baselines3 import DQN

NUM_FEATURES = 48
NUM_ACTIONS = 25
NUM_BINS = 18

C_0 = -0.025
C_1 = -0.125
C_2 = -2
constants = ['age', 'race_white', 'race_black', 'race_hispanic', 'race_other', 'height', 'weight']
FEATURES = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
            'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
            'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
            'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
            'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
            'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
            'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
            'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
            'blood_culture_positive', 'action', 'state_idx']
SOFA_IDX = FEATURES.index("sofa")
LACTATE_IDX = FEATURES.index("LACTATE")


def get_discrete_bin(x):
    if x <= -3:
        return 0
    elif x <= -2:
        return 1
    elif x <= -1.5:
        return 2
    elif x <= -1:
        return 3
    elif x <= -0.8:
        return 4
    elif x <= -0.6:
        return 5
    elif x <= -0.4:
        return 6
    elif x <= -0.2:
        return 7
    elif x <= 0:
        return 8
    elif x < 0.2:
        return 9
    elif x < 0.4:
        return 10
    elif x < 0.6:
        return 11
    elif x < 0.8:
        return 12
    elif x < 1:
        return 13
    elif x < 1.5:
        return 14
    elif x < 2:
        return 15
    elif x < 3:
        return 16
    else:
        return 17


def generate_instances(
        NUM_PATIENTS,
        EFFECT_SIZE,
        discount,
        env_type,
        sample_size=10,
        softness=1,
        demonstrate_softness=100,
        num_trajectories=100,
        seed=0,
        noise_fraction=0,
        feature_size=16,
        noise_dimensions=0,
        START_ADHERING=1,
        data_file=None,
        cf_fraction=0,
        cf_bias=0,
        cf_noise_std=0
):
    print('Generating training instances...')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_labels = []
    full_features = []
    dataset = []

    # Transition Matrix: (NUM_PATIENTS) * (NUM_ACTIONS=25 * NUM_STATES=18^2 * NUM_STATES=18^2)
    num_targets = NUM_PATIENTS
    target_size = 2624400

    policy_kwargs = {'softness': softness, 'net_arch': [32, 32], 'activation_fn': torch.nn.ReLU,
                     'squash_output': False, 'scale': 1}

    # channel_size_list = [label_size, 64, 64, feature_size]
    channel_size_list = [target_size + noise_dimensions, 64, 64, feature_size]
    mlp = MLP(channel_size_list, activation='ReLU', last_activation='Linear')
    mlp.eval()

    for sample_id in tqdm.tqdm(range(sample_size)):
        # Get T_matrices from data
        env = env_type(NUM_PATIENTS=NUM_PATIENTS, EFFECT_SIZE=EFFECT_SIZE, START_ADHERING=START_ADHERING, DATA_PATH=data_file)
        transition_prob = env.true_patient_models.to(torch.float)

        # Add feature space noise and generate features
        if noise_dimensions > 0:
            noisy_transition_prob = torch.cat(
                [transition_prob.view(num_targets, target_size),
                 torch.normal(0, 1, (num_targets, noise_dimensions))], dim=1)
            feature = mlp(noisy_transition_prob.view(num_targets, target_size + noise_dimensions)).detach()
        else:
            feature = mlp(transition_prob.view(num_targets, target_size)).detach()

        # Solve for a good policy
        # TODO: Reset parameter values
        model = DQN('MlpPolicy', env, learning_starts=200, learning_rate=0.0005, target_update_interval=2000,
                    policy_kwargs=policy_kwargs, verbose=2, gamma=discount, strict=False, seed=seed, device=device)
        model.learn(total_timesteps=500, log_interval=100)

        # Generate trajectories using this policy
        obs = env.reset()
        trajectories = []
        trajectory = []
        new_softness = demonstrate_softness

        model.policy.softness = new_softness
        model.policy.q_net.softness = new_softness
        model.policy.q_net_target.softness = new_softness
        while len(trajectories) < num_trajectories:
            # Pick an action
            q_values, probs = model.policy.q_net(obs.view(1, -1))
            # print(q_values, probs)
            # distribution = Categorical(probs)
            # action = distribution.sample()[0] # random action
            # strict action to choose the highest belief
            # action = torch.argmax(q_values)
            action, logprob = model.policy.predict_with_logprob(obs.view(1, -1), deterministic=False)

            # Backup memory for cf
            backup_memory = copy.deepcopy(env.memory[0])
            backup_hidden_state = np.copy(env.hidden_state[0])
            backup_state_0 = np.copy(env.state_0[0]) if env.state_0 is not None else None
            backup_state_idx = env.state_idx

            # Factual step
            obs2, reward, done, info = env.step(action)

            # Generate cf_data for this step
            cf_data = {}
            for cf_action in range(env.action_space.n):
                if cf_action == action.item():
                    continue  # Skip factual action
                if random.random() < cf_fraction:  # Randomly annotate partial cf_actions
                    # Restore backup for cf simulation
                    env.memory[0] = copy.deepcopy(backup_memory)
                    env.hidden_state[0] = np.copy(backup_hidden_state)
                    env.state_0[0] = np.copy(backup_state_0) if backup_state_0 is not None else None
                    env.state_idx = backup_state_idx

                    # Simulate cf: append cf_action to memory
                    env.memory[0].append(
                        np.append(np.append(env.hidden_state[0].reshape((1, NUM_FEATURES - 2)), cf_action), env.state_idx))
                    memory_array = np.expand_dims(env.memory[0], 0)
                    next_state_cf = env.state_model(tf.convert_to_tensor(memory_array[:, :, :-1], dtype=tf.float32),
                                                    training=False).numpy()
                    for constant in constants:
                        idx = FEATURES.index(constant)
                        next_state_cf[0, idx] = env.state_0[0, idx]

                    termination_cf = env.termination_model(tf.convert_to_tensor(memory_array, dtype=tf.float32),
                                                           training=False).numpy()
                    outcome_cf = env.outcome_model(tf.convert_to_tensor(memory_array, dtype=tf.float32),
                                                   training=False).numpy()
                    termination_categories = ['continue', 'done']
                    outcome_categories = ['death', 'release']
                    termination_state_cf = termination_categories[np.argmax(termination_cf)]
                    outcome_state_cf = outcome_categories[np.argmax(outcome_cf)]

                    cf_reward = C_0 * int(next_state_cf[0, SOFA_IDX] == env.hidden_state[0][SOFA_IDX, 0, 0]) + C_1 * (
                                next_state_cf[0, SOFA_IDX] - env.hidden_state[0][SOFA_IDX, 0, 0]) + C_2 * np.tanh(
                        next_state_cf[0, LACTATE_IDX] - env.hidden_state[0][LACTATE_IDX, 0, 0])
                    if termination_state_cf == 'done':
                        cf_reward = -15 if outcome_state_cf == 'death' else 15

                    # Add bias and noise
                    cf_reward += np.random.normal(cf_bias, cf_noise_std)
                    next_state_cf = torch.tensor([get_discrete_bin(
                        next_state_cf.reshape(NUM_FEATURES - 2, 1, 1)[SOFA_IDX, 0, 0]) * NUM_BINS + get_discrete_bin(
                        next_state_cf.reshape(NUM_FEATURES - 2, 1, 1)[LACTATE_IDX, 0, 0])], dtype=torch.int)

                    cf_data[cf_action] = {
                        'reward': cf_reward,
                        'prob': probs[0, cf_action].detach().item(),
                        'next_state': next_state_cf
                    }
                else:
                    cf_data[cf_action] = None  # No annotation for this cf_action

            # Restore factual state after all cf (though not necessary since the factual step is already done)

            trajectory.append((obs.detach(), action.detach(), reward, obs2.detach(), probs.detach(), cf_data))
            obs = obs2
            if done:
                obs = env.reset()
                trajectories.append(trajectory)
                trajectory = []

        full_features.append(feature.view(1, -1, feature_size))
        full_labels.append(transition_prob.flatten().view(1, -1, target_size))
        dataset.append([sample_id, NUM_PATIENTS, EFFECT_SIZE, feature, transition_prob, trajectories])

    full_labels = torch.cat(full_labels, dim=0)
    full_features = torch.cat(full_features, dim=0)

    if sample_size > 1:
        feature_shape = full_features.shape
        full_features = full_features.view(-1, feature_size)
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features * (1 - noise_fraction) + torch.normal(0, 1, full_features.shape) * noise_fraction
        full_features = (full_features - torch.mean(full_features, dim=0)) / (torch.std(full_features, dim=0) + 0.001)
        full_features = full_features.view(feature_shape)

    for i in range(len(dataset)):
        dataset[i][3] = full_features[i]  # update features in dataset

    if cf_fraction > 0:
        with open(f'cf_dataset_seed{seed}_size{sample_size}.pkl', 'wb') as f:
            pickle.dump((dataset, {'feature size': feature_size, 'label size': 1}), f)
        print(f"Saved dataset to cf_dataset_seed{seed}_size{sample_size}.pkl")
    else:
        with open(f'dataset_seed{seed}_size{sample_size}.pkl', 'wb') as f:
            pickle.dump((dataset, {'feature size': feature_size, 'label size': 1}), f)
        print(f"Saved dataset to dataset_seed{seed}_size{sample_size}.pkl")

    print('feature mean {}, std {}'.format(torch.mean(full_features.view(-1, feature_size), dim=0),
                                           torch.std(full_features.view(-1, feature_size), dim=0)))

    print('Finished generating traning instances.')

    return dataset, {'feature size': feature_size, 'label size': 1}


if __name__ == '__main__':
    from gym_sepsis.envs.SepsisWorld import SepsisWorld
    generate_instances(1, 0.2, 0.9, SepsisWorld)