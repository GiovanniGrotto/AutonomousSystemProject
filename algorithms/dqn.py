import os
import time
import random
import wandb
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass
import matplotlib.pyplot as plt
from AutonomousSystemProject.utils import set_seed

from AutonomousSystemProject.GeneralizedOvercooked import GeneralizedOvercooked
from AutonomousSystemProject.algorithms.ReplayBuffer import ReplayBuffer

load_dotenv()  # Loads .env into os.environ
wandb.login(key=os.getenv("WANDB_API_KEY"))


@dataclass
class args:
    seed: int = 1
    cuda: bool = True

    horizon: int = 400
    total_timesteps = 1000000
    num_episodes: int = round(total_timesteps / horizon)
    learning_rate: float = 1e-4
    num_envs: int = 1
    buffer_size: int = 200000#total_timesteps * 0.2
    gamma: float = 0.9
    tau: float = 1
    target_network_frequency: int = 2000
    batch_size: int = 512
    start_e: float = 1
    end_e: float = 0.1
    exploration_fraction: float = 0.7
    learning_starts: int = 80000
    train_frequency: int = 4
    clipnorm: float = 1


class QNetwork(tf.keras.Model):
    def __init__(self, output_len):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.fc3 = tf.keras.layers.Dense(256, activation='relu')
        self.fc4 = tf.keras.layers.Dense(64, activation='relu')
        self.fc5 = tf.keras.layers.Dense(output_len)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@tf.function
def train_step(replay_obs, replay_next_obs, replay_action, replay_reward, replay_done):
    with tf.GradientTape() as tape:
        q_values = q_network(replay_obs)
        q_action = tf.reduce_sum(
            q_values * tf.one_hot(tf.cast(tf.reshape(replay_action, [-1]), tf.int32), output_dim),
            axis=1)

        next_q = target_network(replay_next_obs)
        max_next_q = tf.reduce_max(next_q, axis=1)
        target_q = tf.reshape(replay_reward, [-1]) + args.gamma * max_next_q * (1 - tf.reshape(replay_done, [-1]))

        #loss = tf.reduce_mean(tf.square(target_q - q_action))
        loss = tf.keras.losses.Huber(delta=1.0)(target_q, q_action)

    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    return loss


if __name__ == "__main__":
    set_seed()
    """profiler = cProfile.Profile()
    profiler.enable()"""

    device = '/gpu:0' if args.cuda and tf.config.list_physical_devices('GPU') else '/cpu:0'
    print(device)

    layouts = [
        "cramped_room",
        "asymmetric_advantages",
        "coordination_ring",
        "counter_circuit_o_1order"
    ]
    env = GeneralizedOvercooked(layouts, horizon=args.horizon, use_r_shaped=True, old_dynamics=True)
    model_path = os.path.join(os.getcwd(), f"{', '.join(layouts)}_dqn.weights.h5")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    with tf.device(device):
        # Move network creation to GPU
        q_network = QNetwork(output_dim)
        q_network(tf.convert_to_tensor(np.zeros((1, input_dim)), dtype=tf.float32))  # Build the model on GPU
        try:
            q_network.load_weights(model_path)
            print("Model loaded correctly")
        except FileNotFoundError:
            print("Model to load not found")

        target_network = QNetwork(output_dim)
        target_network(tf.convert_to_tensor(np.zeros((1, input_dim)), dtype=tf.float32))  # Build target network on GPU
        target_network.set_weights(q_network.get_weights())

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)

    # Init wandb
    wandb.init(
        project="overcooked",
        name=f"{q_network.__class__.__name__}_{', '.join(layouts)}_{datetime.now().strftime('%d/%m-%H:%M')}",
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "exploration_fraction": args.exploration_fraction,
            "start_e": args.start_e,
            "end_e": args.end_e,
            "buffer_size": args.buffer_size,
            "train_frequency": args.train_frequency,
            "target_network_frequency": args.target_network_frequency,
            "tau": args.tau,
            "clipnorm": args.clipnorm,
            "total_timesteps": args.total_timesteps,
            "layout": ", ".join(layouts),
            "episode_horizon": args.horizon
        }
    )

    rb = ReplayBuffer(max_size=args.buffer_size, num_envs=args.num_envs, obs_shape=input_dim, action_shape=1)
    rewards_buffer = []
    global_step = 0

    epsilon_schedule = [linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, t)
                        for t in range(args.total_timesteps)]

    loop = tqdm(range(args.num_episodes))
    start = time.time()
    losses = []
    for episode in loop:
        obs = env.reset()
        agents_obs, env_state, _ = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']
        done = False
        total_reward = 0
        ep_len = 0  # DEBUG purpose

        while not done:
            agent1_obs, agent2_obs = agents_obs

            epsilon = epsilon_schedule[global_step]
            if random.random() < epsilon:
                agent1_action = env.action_space.sample()
                agent2_action = env.action_space.sample()
            else:
                with tf.device(device):
                    agents_obs_tensor = tf.convert_to_tensor(agents_obs, dtype=tf.float32)

                    q_vals = q_network(agents_obs_tensor)
                    agent1_action, agent2_action = tf.argmax(q_vals, axis=1).numpy()

            obs, reward, done, env_info = env.step((agent1_action, agent2_action))
            agents_next_obs, env_state, _ = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']

            r1 = env_info['sparse_r_by_agent'][0] + env_info['shaped_r_by_agent'][0]
            r2 = env_info['sparse_r_by_agent'][1] + env_info['shaped_r_by_agent'][1]

            # Add to replay buffer
            rb.add(agent1_obs, agents_next_obs[0], agent1_action, r1, done)
            rb.add(agent2_obs, agents_next_obs[1], agent2_action, r2, done)

            agents_obs = agents_next_obs
            total_reward += (r1 + r2)
            global_step += 1
            ep_len += 1

            avg_rew = 0
            if global_step > args.learning_starts and global_step > args.batch_size:
                avg_rew = np.mean(rewards_buffer[-10:])
                if global_step % args.train_frequency == 0:
                    # Sample batch from replay buffer and move to GPU
                    replay_obs, replay_next_obs, replay_action, replay_reward, replay_done = rb.sample(args.batch_size)
                    with tf.device(device):
                        replay_obs = tf.convert_to_tensor(replay_obs, dtype=tf.float32)
                        replay_next_obs = tf.convert_to_tensor(replay_next_obs, dtype=tf.float32)

                    loss = train_step(replay_obs, replay_next_obs, replay_action, replay_reward, replay_done)
                    losses.append(loss)

                    if global_step % args.target_network_frequency == 0:
                        for target_var, source_var in zip(target_network.trainable_variables, q_network.trainable_variables):
                            target_var.assign(args.tau * source_var + (1.0 - args.tau) * target_var)
                        #target_network.set_weights(q_network.get_weights())

            if global_step % 20000 == 0:
                q_network.save_weights(model_path)

            step_per_second = int(global_step // (time.time() - start))
            loop.set_postfix_str(f"{step_per_second}step/s, avg_reward={avg_rew:.1f}, global_step={global_step}, eps:{epsilon:.3f}, loss={np.mean(losses[-10:]):.4f}")

        if ep_len != args.horizon:
            print(f"EPISODE LEN WAS DIFFERENT FROM HORIZION: {ep_len} != {args.horizon}")

        rewards_buffer.append(total_reward)
        wandb.log({
            "episode_reward": total_reward,
            "avg_reward_10": np.mean(rewards_buffer[-10:]),
            "epsilon": epsilon,
            "loss": np.mean(losses[-10:]) if losses else 0.0,
            "global_step": global_step,
        })

    plt.plot(losses)
    plt.xlabel("Training step (approx)")
    plt.ylabel("Loss")
    plt.title("Q-Network Training Loss Over Time")
    plt.grid(True)
    plt.show()

    plt.plot(rewards_buffer)
    plt.xlabel("Training step (approx)")
    plt.ylabel("Avg Reward")
    plt.title("Q-Network Average Reward Over Time")
    plt.grid(True)
    plt.show()
    """profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(25)  # Show top 25 time consumers"""