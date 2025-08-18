import gym
from gym.utils import seeding
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from collections import deque
import pandas as pd
from gym import spaces

STATE_MODEL = "model/sepsis_states.model"
TERMINATION_MODEL = "model/sepsis_termination.model"
OUTCOME_MODEL = "model/sepsis_outcome.model"
STARTING_STATES_VALUES = "model/sepsis_starting_states.npz"

NUM_FEATURES = 48  # 46 + action + state index
NUM_ACTIONS = 25

EPISODE_MEMORY = 10

C_0 = -0.025
C_1 = -0.125
C_2 = -2

features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
            'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
            'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
            'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
            'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
            'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
            'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
            'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
            'blood_culture_positive', 'action', 'state_idx']

class SepsisEnv(gym.Env):
    """
    Built from trained models on top of the MIMIC dataset, this
    Environment simulates the behavior of the Sepsis patient
    in response to medical interventions.
    For details see: https://github.com/akiani/gym-sepsis 
    """
    metadata = {'render.modes': ['ansi']}

    def __init__(self, starting_state=None, verbose=False):
        super(SepsisEnv, self).__init__()
        module_path = os.path.dirname(__file__)
        self.verbose = verbose
        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL))
        self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL))
        self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL))
        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']
        self.seed()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # use a pixel to represent next state
        self.observation_space = spaces.Box(low=0, high=NUM_ACTIONS, shape=(NUM_FEATURES-2, 1, 1), dtype=np.float32)
        self.reset(starting_state=starting_state)
    
    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque([np.zeros(shape=[NUM_FEATURES])] * 10, maxlen=10)
        if starting_state is None:
            self.s = self.starting_states[np.random.randint(0, len(self.starting_states))][:-1]
        else:
            self.s = starting_state

        self.s = self.s.reshape(NUM_FEATURES - 2, 1, 1)
        self.state_0 = np.copy(self.s)

        if self.verbose:
            print("starting state:", self.s)
        return self.s

    def step(self, action):
        # create memory of present
        self.memory.append(np.append(np.append(self.s.reshape((1, NUM_FEATURES - 2)), action), self.state_idx))
        if self.verbose:
            print("running on memory: ", self.memory)

        memory_array = np.expand_dims(self.memory, 0)
        # next_state = self.state_model.predict(memory_array[:, :, :-1])
        next_state = self.state_model(tf.convert_to_tensor(memory_array[:, :, :-1], dtype=tf.float32), training=False).numpy()

        # overwrite constant variables (these should't change during episode)
        constants = ['age', 'race_white', 'race_black', 'race_hispanic', 'race_other', 'height', 'weight']
        for constant in constants:
            idx = features.index(constant)
            val = self.state_0[idx]
            next_state[0, idx] = val

        # termination = self.termination_model.predict(memory_array)
        termination = self.termination_model(tf.convert_to_tensor(memory_array, dtype=tf.float32), training=False).numpy()
        print(termination)
        # outcome = self.outcome_model.predict(memory_array)
        outcome = self.outcome_model(tf.convert_to_tensor(memory_array, dtype=tf.float32), training=False).numpy()
        print(outcome)

        termination_categories = ['continue', 'done']
        outcome_categories = ['death', 'release']

        termination_state = termination_categories[np.argmax(termination)]
        outcome_state = outcome_categories[np.argmax(outcome)]

        sofa_idx = features.index('sofa')
        lactate_idx = features.index('LACTATE')
        reward = C_0 * int(next_state[0, sofa_idx] == self.s[sofa_idx, 0, 0]) + C_1 * (next_state[0, sofa_idx] - self.s[sofa_idx, 0, 0]) + C_2 * np.tanh(next_state[0, lactate_idx] - self.s[lactate_idx, 0, 0])
        # reward = 0
        done = False

        if termination_state == 'done':
            done = True
            if outcome_state == 'death':
                reward = -15
            else:
                reward = 15

        # keep next state in memory
        self.s = next_state.reshape(46, 1, 1)
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(done)
        return self.s, reward, done, {"prob" : 1}

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='ansi'):
        df = pd.DataFrame(self.memory, columns=features, index=range(0, 10))
        print(df)


class SepsisBeliefEnv(SepsisEnv):
    def __init__(self, starting_state=None, verbose=False):
        super(SepsisEnv, self).__init__()
        module_path = os.path.dirname(__file__)
        self.verbose = verbose
        self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL))
        self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL))
        self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL))
        self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']
        self.seed()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # use a pixel to represent next state
        self.observation_space = spaces.Box(low=0, high=NUM_ACTIONS, shape=(NUM_FEATURES-2, 1, 1), dtype=np.float32)
        self.reset(starting_state=starting_state)

    def step(self, action):
        # create memory of present
        self.memory.append(np.append(np.append(self.s.reshape((1, NUM_FEATURES - 2)), action), self.state_idx))
        if self.verbose:
            print("running on memory: ", self.memory)

        memory_array = np.expand_dims(self.memory, 0)
        # next_state = self.state_model.predict(memory_array[:, :, :-1])
        next_state = self.state_model(tf.convert_to_tensor(memory_array[:, :, :-1], dtype=tf.float32), training=False).numpy()

        # overwrite constant variables (these should't change during episode)
        constants = ['age', 'race_white', 'race_black', 'race_hispanic',
                     'race_other', 'height', 'weight']
        for constant in constants:
            idx = features.index(constant)
            val = self.state_0[idx]
            next_state[0, idx] = val

        # termination = self.termination_model.predict(memory_array)
        termination = self.termination_model(tf.convert_to_tensor(memory_array, dtype=tf.float32), training=False).numpy()
        # outcome = self.outcome_model.predict(memory_array)
        outcome = self.outcome_model(tf.convert_to_tensor(memory_array, dtype=tf.float32), training=False).numpy()

        termination_categories = ['continue', 'done']
        outcome_categories = ['death', 'release']

        termination_state = termination_categories[np.argmax(termination)]
        outcome_state = outcome_categories[np.argmax(outcome)]

        sofa_idx = features.index('sofa')
        lactate_idx = features.index('LACTATE')
        reward = C_0 * int(next_state[0, sofa_idx] == self.s[sofa_idx, 0, 0]) + C_1 * (next_state[0, sofa_idx] - self.s[sofa_idx, 0, 0]) + C_2 * np.tanh(next_state[0, lactate_idx] - self.s[lactate_idx, 0, 0])
        # reward = 0
        done = False

        if termination_state == 'done':
            done = True
            if outcome_state == 'death':
                reward = -15
            else:
                reward = 15

        # keep next state in memory
        self.s = next_state.reshape(46, 1, 1)
        self.state_idx += 1
        self.rewards.append(reward)
        self.dones.append(done)
        return self.s, reward, done, {"prob" : 1}

    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque([np.zeros(shape=[NUM_FEATURES])] * 10, maxlen=10)
        if starting_state is None:
            self.s = self.starting_states[np.random.randint(0, len(self.starting_states))][:-1]
        else:
            self.s = starting_state

        self.s = self.s.reshape(NUM_FEATURES - 2, 1, 1)
        self.state_0 = np.copy(self.s)

        if self.verbose:
            print("starting state:", self.s)
        return self.s

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='ansi'):
        df = pd.DataFrame(self.memory, columns=features, index=range(0, 10))
        print(df)