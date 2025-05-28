import numpy as np
import tensorflow as tf
import random

from AutonomousSystemProject.algorithms.dqn import QNetwork, args
from AutonomousSystemProject.algorithms.ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(self, model_path, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rb = ReplayBuffer(max_size=args.buffer_size, num_envs=args.num_envs, obs_shape=input_dim, action_shape=1)
        self.device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'

        with tf.device(self.device):
            # Move network creation to GPU
            self.q_network = QNetwork(output_dim)
            dummy_input = tf.convert_to_tensor(np.zeros((1, input_dim)), dtype=tf.float32)
            self.q_network(dummy_input)

            try:
                self.q_network.load_weights(model_path)
                print(f"Model {self.q_network.__class__.__name__} loaded correctly")
            except FileNotFoundError:
                print("Model to load not found")

    def action(self, state, eps=0.1):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if random.random() < eps:
            action = random.choice(range(self.output_dim))
            q_vals = [0] * self.output_dim
            q_vals[action] = 1
        else:
            if len(state.shape) == 1:
                state = tf.expand_dims(state, axis=0)  # Add batch dimension if missing
            q_vals = self.q_network(state).numpy().squeeze()
            action = np.argmax(q_vals)
        return action, q_vals

    def actions(self, states):
        with tf.device(self.device):
            agents_obs_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            q_vals = self.q_network(agents_obs_tensor)
            actions = tf.argmax(q_vals, axis=1).numpy()
        return actions
