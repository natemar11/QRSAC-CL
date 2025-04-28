from collections import OrderedDict

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.env_utils import get_dim
from rlkit.data_management.replay_buffer import ReplayBuffer


class TorchReplayBuffer(ReplayBuffer):

    def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
        # new: support multi-dimensional observations (H×W×C)
        observation_shape = env.observation_space.shape
        # for any legacy code that still needs a flat dim
        observation_dim = int(np.prod(observation_shape))
        action_dim = get_dim(env.action_space)

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        self._max_replay_buffer_size = max_replay_buffer_size

        # new: allocate with full image shape
        self._observations = torch.zeros(
            (max_replay_buffer_size, *observation_shape),
            dtype=torch.float,
        )
        self._next_obs = torch.zeros(
            (max_replay_buffer_size, *observation_shape),
            dtype=torch.float,
        )

        self._actions   = torch.zeros((max_replay_buffer_size, action_dim), dtype=torch.float)
        self._rewards   = torch.zeros((max_replay_buffer_size, 1),          dtype=torch.float)
        self._terminals = torch.zeros((max_replay_buffer_size, 1),          dtype=torch.float)

        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = torch.zeros((max_replay_buffer_size, size), dtype=torch.float)
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0

        if ptu.gpu_enabled():
            # self.stream = torch.cuda.Stream(ptu.device)
            self.batch = None

    def add_sample(self, observation, action, reward, next_observation, terminal, env_info, **kwargs):
        # assign the full image (casts to float to match buffer dtype)
        self._observations[self._top] = torch.from_numpy(observation).float()
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        self._actions[self._top] = torch.from_numpy(action)
        self._rewards[self._top] = torch.from_numpy(reward)
        self._terminals[self._top] = torch.from_numpy(terminal)
        self._next_obs[self._top] = torch.from_numpy(next_observation).float()

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = torch.from_numpy(env_info[key])
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        #As per GT Sophy  paper, n-step look ahead with n=7
        indices = np.random.randint(0, self._size-7, batch_size)
        batch = dict(
            indices = torch.from_numpy(indices),
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    
    def nstep_batch(self, n, index):
            indices = np.arange(index, index + n) % self._size
            nstep_rewards = self._rewards[indices]
            return nstep_rewards

    
    def get_n_step_data(self, n, indices):
            indices_list = indices.cpu().numpy()
            indices_list_plus_n = indices_list + n
            # nstep_rewards = self._rewards[indices]
            batch = dict(
                observations=self._observations[indices_list_plus_n],
                actions=self._actions[indices_list_plus_n],
                rewards=self._rewards[indices_list_plus_n],
                terminals=self._terminals[indices_list_plus_n],
                next_observations=self._next_obs[indices_list_plus_n],
            )
            for key in self._env_info_keys:
                assert key not in batch.keys()
                batch[key] = self._env_infos[key][indices_list_plus_n]
            return batch


    def preload(self, batch_size):
        try:
            self.batch = self.random_batch(batch_size)
        except StopIteration:
            self.batch = None
            return
        if ptu.gpu_enabled():
            # with torch.cuda.stream(self.stream):
            for k in self.batch:
                self.batch[k] = self.batch[k].to(device=ptu.device, non_blocking=True)

    def next_batch(self, batch_size):
        # torch.cuda.current_stream(ptu.device).wait_stream(self.stream)
        if self.batch is None:
            self.preload(batch_size)
        batch = self.batch
        self.preload(batch_size)
        return batch

    def rebuild_env_info_dict(self, idx):
        return {key: self._env_infos[key][idx] for key in self._env_info_keys}

    def batch_env_info_dict(self, indices):
        return {key: self._env_infos[key][indices] for key in self._env_info_keys}

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([('size', self._size)])
