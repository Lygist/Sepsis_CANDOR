import gym
import torch
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from collections import deque
from gym import spaces
from gym_sepsis.envs.SepsisWorldUtils import *
from gym.utils import seeding


STATE_MODEL = "model/sepsis_states.model"
TERMINATION_MODEL = "model/sepsis_termination.model"
OUTCOME_MODEL = "model/sepsis_outcome.model"
STARTING_STATES_VALUES = "model/sepsis_starting_states.npz"

NUM_FEATURES = 48  # 46 + action + state index
NUM_ACTIONS = 25
NUM_BINS = 18

EPISODE_MEMORY = 10

C_0 = -0.025
C_1 = -0.125
C_2 = -2

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

class SepsisWorld(gym.Env):
    def __init__(
        self,
        transition_prob=None,  # predictions for the transition matrix
        true_transition_prob=None,  # for forward compatability with TBBeliefWorld
        DATA_PATH=None,  # location of file with patient data
        NUM_PATIENTS=1,  # number of patients to sample (ALWAYS ASSUMED TO BE 1 FOR THIS PROJECT)
        MIN_SEQ_LEN=30,  # Data pre-processing step
        EFFECT_SIZE=0.2,  # size of action effect
        EPSILON=0.05,  # smoothing for transition probabilities
        EPISODE_LEN=10,  # how many steps consist of an episode
        START_ADHERING=1,  # whether patients start in an adhering state or not
    ):
        super(SepsisWorld, self).__init__()

        # SAVE IMPORTANT PARAMETERS
        self.NUM_PATIENTS = NUM_PATIENTS
        self.EPISODE_LEN = EPISODE_LEN

        # DEFINE STATE SPACE
        self.observation_space = spaces.Box(low=0, high=NUM_BINS, shape=(NUM_PATIENTS,), dtype=np.int32)

        # DEFINE ACTION SPACE
        self.action_space = spaces.Discrete(NUM_PATIENTS * NUM_ACTIONS)

        # DEFINE TRANSITIONS
        # patient_models[patient][action][current_state][next_state] --> probability of event
        if true_transition_prob is not None:
            self.true_patient_models = true_transition_prob
        elif transition_prob is not None:
            self.true_patient_models = transition_prob
        else:
            with open("original_trajectories.pkl", "rb") as f:
                original_trajectories = pickle.load(f)
            patient_model = np.zeros((NUM_ACTIONS, NUM_BINS ** 2, NUM_BINS ** 2))
            for trajectory in original_trajectories:
                for transition in trajectory:
                    current_state = int(get_discrete_bin(transition[0][SOFA_IDX, 0, 0]) * NUM_BINS + get_discrete_bin(transition[0][LACTATE_IDX, 0, 0]))
                    next_state = int(get_discrete_bin(transition[3][SOFA_IDX, 0, 0]) * NUM_BINS + get_discrete_bin(transition[3][LACTATE_IDX, 0, 0]))
                    patient_model[transition[1]][current_state][next_state] += 1
            totals = np.sum(patient_model, axis=2, keepdims=True)
            totals[totals == 0] = 1
            patient_model = patient_model / totals
            print(patient_model.mean(), (patient_model == 0).mean())
            self.true_patient_models = torch.from_numpy(np.stack([patient_model] * NUM_PATIENTS, axis=0)).float()

        # LOAD MODELS
        module_path = os.path.dirname(__file__)
        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL))
        self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL))
        self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL))
        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']

        # Initialise internal variables
        self.reset()

    def reset(self):
        self.rewards = [[]] * self.NUM_PATIENTS
        self.dones = [[]] * self.NUM_PATIENTS
        self.state_idx = 0
        self.current_step = 0
        self.memory = [deque([np.zeros(shape=[NUM_FEATURES])] * 10, maxlen=10)] * self.NUM_PATIENTS
        start_states = self.starting_states[np.random.choice(len(self.starting_states), self.NUM_PATIENTS, replace=False)]
        # start_states = np.random.choice(self.starting_states, self.NUM_PATIENTS, replace=False)
        self.hidden_state = np.stack([start_state[:-1].reshape(NUM_FEATURES - 2, 1, 1) for start_state in start_states], axis=0)
        self.state_0 = np.copy(self.hidden_state)
        self.state = torch.tensor([get_discrete_bin(start_state[:-1].reshape(NUM_FEATURES - 2, 1, 1)[SOFA_IDX, 0, 0]) * NUM_BINS + get_discrete_bin(start_state[:-1].reshape(NUM_FEATURES - 2, 1, 1)[LACTATE_IDX, 0, 0]) for start_state in start_states], dtype=torch.int)
        self.terminal_state = False
        return self.observe()

    def is_terminal(self):
        return (self.current_step >= self.EPISODE_LEN)

    def observe(self):
        return self.state

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]

    def _get_action_onehot(self, action):
        if not torch.is_tensor(action):
            action = torch.tensor([action], dtype=torch.long)
        action_onehot = torch.nn.functional.one_hot(action, num_classes=self.NUM_PATIENTS * NUM_ACTIONS).flatten()
        return action_onehot

    def _get_next_state(self, action_onehot):
        action = torch.argmax(action_onehot).item()
        self.memory[0].append(np.append(np.append(self.hidden_state[0].reshape((1, NUM_FEATURES - 2)), action), self.state_idx))
        memory_array = np.expand_dims(self.memory[0], 0)
        next_state = self.state_model(tf.convert_to_tensor(memory_array[:, :, :-1], dtype=tf.float32), training=False).numpy()

        # overwrite constant variables (these should't change during episode)
        constants = ['age', 'race_white', 'race_black', 'race_hispanic', 'race_other', 'height', 'weight']
        for constant in constants:
            idx = FEATURES.index(constant)
            val = self.state_0[0, idx]
            next_state[0, idx] = val
        
        termination = self.termination_model(tf.convert_to_tensor(memory_array, dtype=tf.float32), training=False).numpy()
        outcome = self.outcome_model(tf.convert_to_tensor(memory_array, dtype=tf.float32), training=False).numpy()

        termination_categories = ['continue', 'done']
        outcome_categories = ['death', 'release']
        termination_state = termination_categories[np.argmax(termination)]
        outcome_state = outcome_categories[np.argmax(outcome)]

        reward = C_0 * int(next_state[0, SOFA_IDX] == self.hidden_state[0][SOFA_IDX, 0, 0]) + C_1 * (next_state[0, SOFA_IDX] - self.hidden_state[0][SOFA_IDX, 0, 0]) + C_2 * np.tanh(next_state[0, LACTATE_IDX] - self.hidden_state[0][LACTATE_IDX, 0, 0])
        done = False

        if termination_state == 'done':
            done = True
            if outcome_state == 'death':
                reward = -15
            else:
                reward = 15
        self.terminal_state = done

        # keep next state in memory
        self.hidden_state[0] = next_state.reshape(46, 1, 1)
        self.state_idx += 1
        self.rewards[0].append(reward)
        self.dones[0].append(done)
        
        next_state = torch.tensor([get_discrete_bin(next_state.reshape(NUM_FEATURES - 2, 1, 1)[SOFA_IDX, 0, 0]) * NUM_BINS + get_discrete_bin(next_state.reshape(NUM_FEATURES - 2, 1, 1)[LACTATE_IDX, 0, 0])], dtype=torch.int)

        next_state_probabilities = torch.cat([self.true_patient_models[i, action_onehot[i], self.state[i]].view(1, -1) for i in range(self.NUM_PATIENTS)], dim=0)
        next_state_probabilities[next_state_probabilities == 0] = 1e-6
        logprob = torch.log(torch.gather(next_state_probabilities, 1, next_state.view(-1, 1).to(dtype=torch.int64))).sum()

        return next_state, logprob, reward

    def step(self, action):
        action_onehot = self._get_action_onehot(action)
        next_state, logprob, reward = self._get_next_state(action_onehot)
        self.state = next_state
        self.current_step += 1
        return next_state, reward, self.is_terminal(), {'logprob': logprob}

    @staticmethod
    def loss_fn(data, transition_prob):
        if isinstance(data[0], list):  # nested trajectories
            N = len(data)
            T = len(data[0])
            states = torch.cat([data[n][t][0].view(1, -1) for n in range(N) for t in range(T)], dim=0).to(torch.int64)
            next_states = torch.cat([data[n][t][3].view(1, -1) for n in range(N) for t in range(T)], dim=0).to(torch.int64)
            actions = torch.cat([data[n][t][1] for n in range(N) for t in range(T)], dim=0)
        else:  # flat D_plus list of tuples
            states = torch.tensor([sample[0].item() for sample in data], dtype=torch.int64)
            next_states = torch.tensor([sample[3].item() for sample in data], dtype=torch.int64)
            actions = torch.tensor([sample[1].item() for sample in data], dtype=torch.int64)

        NUM_PATIENTS = 1  # fixed
        actions_onehot = torch.nn.functional.one_hot(actions, num_classes=NUM_PATIENTS * NUM_ACTIONS)

        # Get relevant probabilities
        likelihoods = transition_prob[torch.zeros_like(actions), actions, states, next_states].unsqueeze(1)  # since i=0
        likelihoods[likelihoods == 0] = 1e-6
        NLL = (-torch.log(likelihoods)).sum() / len(states)
        return NLL

if __name__ == '__main__':
    import random

    # Test Environments
    envs = [SepsisWorld()]

    for env in envs:
        print(f"\nTesting {env.__class__.__name__}:")

        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            print(f"State: {obs.tolist()}, Action: {action}, Reward: {reward}, Next State: {next_obs.tolist()}")

            obs = next_obs