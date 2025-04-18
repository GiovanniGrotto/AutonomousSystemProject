from tqdm import tqdm
from GeneralizedOvercooked import GeneralizedOvercooked
from AutonomousSystemProject.agents.DQNAgent import DQNAgent
from AutonomousSystemProject.agents.RandomAgent import RandomAgent
import tensorflow as tf
import numpy as np
from utils import get_q_values_plt, save_video_from_images, save_img_list, interpret_state
import cv2

"""
import numpy as np
x = np.array(
[ 1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0, -2,  2,  0,  0,  0,  0,
  0,  2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,
 -1, -1,  0,  0,  0,  1,  0,  0,  0,  0,  2,  1,  0,  0,  1,  1,  0,  0,
  0,  0,  0,  0,  1, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
  0,  1, -2,  1,  3,  1,])
y = agent1.action(x)
y -> (2, [6.6403193 6.6514626 6.7327075 6.640445  6.6257725 6.613402 ])
 
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = agent1.action(x)
y -> (2, [ 0.27887285  0.04977222  0.49237794  0.36571524 -0.07192153 -0.38969436])"""
"""
import numpy as np
initial_states = []

for _ in range(6):
    obs = env.reset()
    agents_obs, env_state, _ = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']
    initial_states.append(agents_obs[0])
    initial_states.append(agents_obs[1])
    
stacked = np.stack(initial_states)

# Use np.unique along axis=0 to find unique rows (unique sequences)
unique_sequences = np.unique(stacked, axis=0)

print(f"Number of unique sequences: {len(unique_sequences)}")
print(unique_sequences)
Number of unique sequences: 2
[[ 1.  0.  0.  0.  0.  0.  0.  0. -1. -1.  0.  0.  0.  1.  0.  0.  0.  0.
   2.  1.  0.  0.  1.  1.  0.  0.  0.  0.  0.  0.  1. -2.  0.  0.  0.  0.
   0.  0.  0.  0.  0.  0.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  0.  0.
   1.  0.  0.  0. -2.  2.  0.  0.  0.  0.  0.  2.  0.  0.  1.  1.  0.  0.
   0.  0.  0.  0. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.
   1.  0.  2. -1.  1.  2.]
 [ 1.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0. -2.  2.  0.  0.  0.  0.
   0.  2.  0.  0.  1.  1.  0.  0.  0.  0.  0.  0. -1. -1.  0.  0.  0.  0.
   0.  0.  0.  0.  0.  0.  1.  0.  1.  0.  1.  0.  0.  0.  0.  0.  0.  0.
  -1. -1.  0.  0.  0.  1.  0.  0.  0.  0.  2.  1.  0.  0.  1.  1.  0.  0.
   0.  0.  0.  0.  1. -2.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
   0.  1. -2.  1.  3.  1.]]
"""


BASE_DIR = "evaluations"


def evaluate(env, agent1, agent2, n_episodes=5, render=False, save_traj=False):
    rewards_buff = []
    loop = tqdm(range(n_episodes))
    for i in loop:
        obs = env.reset()
        agents_obs, env_state, _ = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']
        done = False
        tot_reward = 0
        img_buff = []

        while not done:
            agent1_state, agent2_state = tf.convert_to_tensor(agents_obs, dtype=tf.float32)

            assert np.allclose(agents_obs[0], agent1_state.numpy())

            agent1_action, q1_vals = agent1.action(agent1_state)
            agent2_action, q2_vals = agent2.action(agent2_state)

            if render:
                env.render(env_state)
                interpret_state(agent1_state, 0)
                interpret_state(agent2_state, 1)

            if save_traj:
                state_img = env.get_state_img(env_state)
                q_val_img = get_q_values_plt([q1_vals, q2_vals])
                height = state_img.shape[0]
                state_img = cv2.resize(state_img, (state_img.shape[1], height))
                q_val_img = cv2.resize(q_val_img, (q_val_img.shape[1], height))
                final_img = cv2.hconcat([state_img, q_val_img])
                img_buff.append(final_img)

            obs, reward, done, env_info = env.step((agent1_action, agent2_action))
            agents_obs, env_state, _ = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']
            tot_reward += reward

        rewards_buff.append(tot_reward)
        loop.set_postfix({"avg_rew": np.mean(rewards_buff)})
        save_img_list(img_buff, f"{BASE_DIR}/images_{i}")
        save_video_from_images(img_buff, f'{BASE_DIR}/trajectory_{i}.avi', fps=1)

    return rewards_buff


if __name__ == "__main__":
    #set_seed()
    layouts = ["asymmetric_advantages"]
    horizon = 400
    env = GeneralizedOvercooked(layouts, horizon=horizon, old_dynamics=True)

    dqn_agent = DQNAgent("algorithms/dqn_model.weights.h5", env.observation_space.shape[0], env.action_space.n)
    random_agent = RandomAgent(env.action_space.n)
    print(evaluate(env, dqn_agent, dqn_agent, render=True))


