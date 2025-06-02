from tqdm import tqdm
from utils.GeneralizedOvercooked import GeneralizedOvercooked
from agents.DQNAgent import DQNAgent
from agents.RandomAgent import RandomAgent
from agents.PPOAgent import PPOAgent
from agents.RecPPOAgent import RecPPOAgent
import numpy as np
from utils.utils import get_action_values_plt, save_video_from_images, save_img_list, interpret_state, set_seed
import cv2
import pandas as pd
from itertools import combinations_with_replacement
import os

# Optional: reset pandas display options before printing
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Prevent line wrapping
pd.set_option('display.max_colwidth', None)  # Show full column contents


BASE_DIR = "../evaluations"
os.makedirs(os.path.join(os.getcwd(), BASE_DIR), exist_ok=True)


def evaluate(env, agent1, agent2, n_episodes=5, render=False, save_traj=False):
    rewards_buff = []
    shaped_rew_buff = []
    loop = tqdm(range(n_episodes), desc=f"Evaluting: {agent1.__class__.__name__} - {agent2.__class__.__name__} pair")
    for i in loop:
        obs, _ = env.reset()
        agents_obs, env_state, other_agent_env_idx = obs['both_agent_obs'], obs['overcooked_state'], obs['other_agent_env_idx']
        done = False
        tot_reward = 0
        shaped_tot_reward = 0
        img_buff = []

        if "RecPPOAgent" == agent1.__class__.__name__:
            agent1.init_hidden_state()
        if "RecPPOAgent" == agent2.__class__.__name__:
            agent2.init_hidden_state()

        while not done:
            agent1_state, agent2_state = agents_obs

            agent1_action, action1_vals = agent1.action(agent1_state)
            agent2_action, action2_vals = agent2.action(agent2_state)

            if render:
                env.render(env_state)
                interpret_state(agent1_state, 1-other_agent_env_idx)
                interpret_state(agent2_state, other_agent_env_idx)

            if save_traj:
                state_img = env.get_state_img(env_state)
                actions_val_img = get_action_values_plt([action1_vals, action2_vals], agent_idx=1-other_agent_env_idx)
                height = state_img.shape[0]
                state_img = cv2.resize(state_img, (state_img.shape[1], height))
                actions_val_img = cv2.resize(actions_val_img, (actions_val_img.shape[1], height))
                final_img = cv2.hconcat([state_img, actions_val_img])
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
        if save_traj:
            save_img_list(img_buff, f"{BASE_DIR}/images_{i}")
            save_video_from_images(img_buff, f'{BASE_DIR}/trajectory_{i}.avi', fps=1)

    return rewards_buff, shaped_rew_buff


def evaluate_all(agent_list, layout_list, n_episodes=5, render=False, save_traj=False):
    results = []

    for layout in layout_list:
        env = GeneralizedOvercooked([layout], horizon=400, old_dynamics=True, use_r_shaped=True)
        print(f"\nEvaluating on: {layout}")
        for agent1, agent2 in combinations_with_replacement(agent_list, r=2):
            rewards_buff, shaped_rew_buff = evaluate(
                env=env,
                agent1=agent1,
                agent2=agent2,
                n_episodes=n_episodes,
                render=render,
                save_traj=save_traj
            )

            result = {
                'layout': layout,
                'agent1': agent1.__class__.__name__,
                'agent2': agent2.__class__.__name__,
                'avg_reward': np.mean(rewards_buff),
                'avg_shaped_reward': np.mean(shaped_rew_buff)
            }

            results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    layouts = ["asymmetric_advantages"]
    horizon = 400
    env = GeneralizedOvercooked(layouts, horizon=horizon, old_dynamics=True, use_r_shaped=True)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    random_agent = RandomAgent(env.action_space.n)
    dqn_agent = DQNAgent("../algorithms/model_to_test/dqn/asymmetric_advantages_dqn.weights.h5", input_dim, output_dim)
    ppo_agent = PPOAgent("../algorithms/model_to_test/ppo/cramped_room.pth", input_dim, output_dim)
    rec_ppo_agent_1 = RecPPOAgent("../algorithms/saved_models/PPO-LSTM_asymmetric_advantages_30-05_20-27.pth", input_dim, output_dim)
    rec_ppo_agent_2 = RecPPOAgent("../algorithms/saved_models/PPO-LSTM_asymmetric_advantages_30-05_20-27.pth", input_dim, output_dim)
    print(evaluate(env, rec_ppo_agent_1, rec_ppo_agent_2, save_traj=True, render=False))
    """print(
        evaluate_all(
            agent_list=[random_agent, dqn_agent, ppo_agent, rec_ppo_agent_1, rec_ppo_agent_2],
            layout_list=[
             "cramped_room"
             ],
            save_traj=False
        )
    )"""

