from collections import namedtuple, deque
import numpy as np
import os
import gzip
import pickle

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, history_length=1e5):
        self._data = deque([], maxlen=history_length)
        # self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        # self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

    def push(self, *args):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.append(Transition(*args))

    # def add_transition(self, state, action, next_state, reward, done):
        # self._data.states.append(state)
        # self._data.actions.append(action)
        # self._data.next_states.append(next_state)
        # self._data.rewards.append(reward)
        # self._data.dones.append(done)

    def sample(self, batch_size):
        """
        This method samples a batch of transitions
        """
        return np.random.sample(self._data, batch_size)
    # def next_batch(self, batch_size):
    #     """
    #     This method samples a batch of transitions.
    #     """
    #     batch_indices = np.random.choice(len(self._data), batch_size)
    #     batch_states = np.array([self._data[i][0] for i in batch_indices])
    #     batch_actions = np.array([self._data.actions[i][1] for i in batch_indices])
    #     batch_next_states = np.array([self._data.next_states[i][2] for i in batch_indices])
    #     batch_rewards = np.array([self._data.rewards[i][3] for i in batch_indices])
    #     batch_dones = np.array([self._data.dones[i][4] for i in batch_indices])

        # batch_indices = np.random.choice(len(self._data.states), batch_size)
        # batch_states = np.array([self._data.states[i] for i in batch_indices])
        # batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        # batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        # batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        # batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

    def __len__(self):
        return len(self._data)
