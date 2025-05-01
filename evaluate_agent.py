from tqdm import tqdm
from GeneralizedOvercooked import GeneralizedOvercooked
from AutonomousSystemProject.agents.DQNAgent import DQNAgent
from AutonomousSystemProject.agents.RandomAgent import RandomAgent
import tensorflow as tf
import numpy as np
from utils import get_q_values_plt, save_video_from_images, save_img_list, interpret_state, set_seed
import cv2

"""
import numpy as np
x = np.array(
[ 0.  0.  0.  1.  1.  0.  0.  0.  0.  0.  0.  0. -1.  0.  0.  0.  0.  0.,  1.  2.  0.  0.  1.  1.  0.  0.  0.  0.  0.  0.  3. -1.  1.  1.  0.  0.,  0.  0.  0.  0.  2. -2.  0.  0.  1.  1.  0.  0.  0.  1.  0.  0.  0.  0., -3.  1.  0.  0. -3.  0.  0.  0.  0. )
y = agent1.action(x)
y ->  [2.5636733 2.231389  2.2966123 2.1597965 2.3345888 2.054875 ]

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = agent1.action(x)
y -> (2, [ 0.27887285  0.04977222  0.49237794  0.36571524 -0.07192153 -0.38969436])"""


BASE_DIR = "evaluations"


def evaluate(env, agent1, agent2, n_episodes=5, render=False, save_traj=False):
    rewards_buff = []
    shaped_rew_buff = []
    loop = tqdm(range(n_episodes))
    for i in loop:
        obs = env.reset()
        agents_obs, env_state, other_agent_env_idx = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']
        done = False
        tot_reward = 0
        shaped_tot_reward = 0
        img_buff = []

        while not done:
            agent1_state, agent2_state = tf.convert_to_tensor(agents_obs, dtype=tf.float32)

            assert np.allclose(agents_obs[0], agent1_state.numpy())

            agent1_action, q1_vals = agent1.action(agent1_state)
            agent2_action, q2_vals = agent2.action(agent2_state)

            if render:
                env.render(env_state)
                interpret_state(agent1_state, 1-other_agent_env_idx)
                interpret_state(agent2_state, other_agent_env_idx)

            if save_traj:
                state_img = env.get_state_img(env_state)
                q_val_img = get_q_values_plt([q1_vals, q2_vals], agent1_idx=1-other_agent_env_idx)
                height = state_img.shape[0]
                state_img = cv2.resize(state_img, (state_img.shape[1], height))
                q_val_img = cv2.resize(q_val_img, (q_val_img.shape[1], height))
                final_img = cv2.hconcat([state_img, q_val_img])
                img_buff.append(final_img)

            obs, reward, done, env_info = env.step((agent1_action, agent2_action))
            agents_obs, env_state, other_agent_env_idx = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']
            r1 = env_info['sparse_r_by_agent'][0] + env_info['shaped_r_by_agent'][0]
            r2 = env_info['sparse_r_by_agent'][1] + env_info['shaped_r_by_agent'][1]
            tot_reward += reward
            shaped_tot_reward += r1 + r2

        rewards_buff.append(tot_reward)
        shaped_rew_buff.append(shaped_tot_reward)
        loop.set_postfix({"avg_rew": np.mean(rewards_buff), "avg_shaped_rew": np.mean(shaped_rew_buff)})
        save_img_list(img_buff, f"{BASE_DIR}/images_{i}")
        save_video_from_images(img_buff, f'{BASE_DIR}/trajectory_{i}.avi', fps=1)

    return rewards_buff, shaped_rew_buff


# Remember that dqn train use shaped reward while eval use sparse reward
if __name__ == "__main__":
    set_seed()
    layouts = ["asymmetric_advantages"]
    horizon = 400
    env = GeneralizedOvercooked(layouts, horizon=horizon, old_dynamics=True, use_r_shaped=True)

    dqn_agent = DQNAgent("algorithms/asymmetric_advantages_dqn.weights.h5", env.observation_space.shape[0], env.action_space.n)
    random_agent = RandomAgent(env.action_space.n)
    print(evaluate(env, dqn_agent, dqn_agent, save_traj=True, render=False))


