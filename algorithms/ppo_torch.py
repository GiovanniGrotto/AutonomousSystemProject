import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
import tyro
import numpy as np
from tqdm import tqdm
import time
import random
import os

from AutonomousSystemProject.GeneralizedOvercooked import GeneralizedOvercooked


@dataclass
class Args:
    horizon: int = 400
    cuda: bool = True
    total_timesteps: int = 500000
    learning_rate: float = 1e-4
    num_envs: int = 1
    num_steps: int = 2048  # longer rollouts help reduce variance
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4  # fewer due to smaller batch size
    update_epochs: int = 5  # slightly more to stabilize updates
    clip_coef: float = 0.2
    ent_coef: float = 0.02  # encourages exploration in sparse-reward env
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = num_steps * num_envs
    minibatch_size: int = batch_size // num_minibatches
    num_iterations: int = total_timesteps // batch_size


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, output_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    set_seed()

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print(device)

    layouts = [
        "cramped_room",
    ]
    env = GeneralizedOvercooked(layouts, horizon=args.horizon, use_r_shaped=True, old_dynamics=True)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = Agent(input_dim, output_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)

    #  This is faster than using a RolloutBuffer since it stay in the GPU
    obs = torch.zeros((args.num_steps, args.num_envs) + (input_dim,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    tot_reward = 0
    reward_buff = []
    loop = tqdm(range(args.num_iterations))
    avg_rew = 0
    loss = 0
    start_time = time.time()

    next_obs, _ = env.reset()
    agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
    agent_idx = 1 - next_obs['other_agent_env_idx']
    next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
    next_done = False
    for i in loop:
        for step in range(args.num_steps):
            if next_done:
                next_obs, _ = env.reset()
                agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
                agent_idx = 1 - next_obs['other_agent_env_idx']
                next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
                reward_buff.append(tot_reward)
                tot_reward = 0
                avg_rew = np.mean(reward_buff[-10:])

            global_step += args.num_envs
            obs[step] = next_obs[agent_idx]
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            actions[step] = action[agent_idx]
            logprobs[step] = logprob[agent_idx]
            values[step] = value[agent_idx]

            agent1_action = action[0].cpu().item()
            agent2_action = action[1].cpu().item()

            next_obs, reward, next_done, env_info = env.step((agent1_action, agent2_action))
            r1 = env_info['sparse_r_by_agent'][0] + env_info['shaped_r_by_agent'][0]
            r2 = env_info['sparse_r_by_agent'][1] + env_info['shaped_r_by_agent'][1]
            tot_reward += (r1 + r2)

            rewards[step] = (r1 + r2)

            agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
            next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)

            step_per_second = int(global_step // (time.time() - start_time))
            loop.set_postfix_str(
                f"{step_per_second}step/s, avg_reward={avg_rew:.1f}, global_step={global_step}, loss={loss}")

        #  Rollout finished, advantage calculation
        with torch.no_grad():
            #  If we don't apply the reset logic at the beginning of the for loop, but we put at the end
            #  we risk to have here an initial state instead of the terminal one here in case we ended in the terminal
            next_value = agent.get_value(next_obs[agent_idx])  # Consider the value only for the first agent
            adv = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
                adv[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
            returns = adv + values

        #  Net update
        b_obs = obs.reshape((-1,) + (input_dim, ))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = adv.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()



