import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, num_envs, obs_shape, action_shape):
        max_size = int(max_size)
        self.max_size = max_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (obs_shape,)
        self.action_shape = action_shape if isinstance(action_shape, tuple) else (action_shape,)
        self.buffer_pos = 0
        self.full = False

        shape = (max_size, num_envs) + self.obs_shape
        self.observations = np.zeros(shape, dtype=np.float32)
        self.next_observations = np.zeros(shape, dtype=np.float32)
        self.actions = np.zeros((max_size, num_envs) + self.action_shape, dtype=np.float32)
        self.rewards = np.zeros((max_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((max_size, num_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.buffer_pos] = obs
        self.next_observations[self.buffer_pos] = next_obs
        self.actions[self.buffer_pos] = action
        self.rewards[self.buffer_pos] = reward
        self.dones[self.buffer_pos] = done

        self.buffer_pos += 1
        if self.buffer_pos == self.max_size:
            self.buffer_pos = 0
            self.full = True

    def sample(self, batch_size):
        upper_bound = self.max_size if self.full else self.buffer_pos
        batch_indexes = np.random.randint(0, upper_bound, size=batch_size)
        env_indexes = np.random.randint(0, self.num_envs, size=batch_size)

        obs_sample = self.observations[batch_indexes, env_indexes]
        next_obs_sample = self.next_observations[batch_indexes, env_indexes]
        actions_sample = self.actions[batch_indexes, env_indexes]
        rewards_sample = self.rewards[batch_indexes, env_indexes]
        dones_sample = self.dones[batch_indexes, env_indexes]

        return obs_sample, next_obs_sample, actions_sample, rewards_sample, dones_sample
