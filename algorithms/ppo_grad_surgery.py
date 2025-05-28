import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict, deque
import time

from AutonomousSystemProject.algorithms.ppo import compute_multiagent_gae, PPOActorCritic
from AutonomousSystemProject.GeneralizedOvercooked import GeneralizedOvercooked


@dataclass
class Args:
    horizon: int = 400
    cuda: bool = True
    total_timesteps: int = 1000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_agents: int = 2
    num_steps: int = 1024  # longer rollouts help reduce variance
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01  # encourages exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 5
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            layer_init(nn.Linear(dim, dim)),
            nn.ReLU(),
            layer_init(nn.Linear(dim, dim)),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class PPOActorCriticResidual(nn.Module):
    def __init__(self, input_dim=96, output_dim=6):
        super().__init__()

        hidden_dim = 256

        # Shared feature extractor
        self.shared = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # Actor head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 128)),
            nn.ReLU(),
            nn.LayerNorm(128),
            layer_init(nn.Linear(128, output_dim), std=0.01)
        )

        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 128)),
            nn.ReLU(),
            nn.LayerNorm(128),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def get_action_and_value(self, obs, action=None):
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def get_value(self, obs):
        features = self.shared(obs)
        return self.critic(features)


def collect_rollouts(env, layout_id, agent):
    input_dim = env.observation_space.shape[0]
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    obs = torch.zeros((args.num_steps, args.num_agents) + (input_dim,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_agents)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_agents)).to(device)

    next_obs, _ = env.reset(layout_id=layout_id)
    agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
    agent_idx = 0
    other_agent_idx = 1
    next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
    next_done = False
    final_rew = 0
    tot_rew = 0

    for step in range(args.num_steps):
        if next_done:
            next_obs, _ = env.reset(layout_id=layout_id)
            agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs[
                'other_agent_env_idx']
            next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
            final_rew = tot_rew
            tot_rew = 0

        obs[step, agent_idx] = next_obs[agent_idx]
        obs[step, other_agent_idx] = next_obs[other_agent_idx]
        dones[step, agent_idx] = next_done
        dones[step, other_agent_idx] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        actions[step, agent_idx] = action[agent_idx]
        actions[step, other_agent_idx] = action[other_agent_idx]
        logprobs[step, agent_idx] = logprob[agent_idx]
        logprobs[step, other_agent_idx] = logprob[other_agent_idx]
        values[step, agent_idx] = value[agent_idx]
        values[step, other_agent_idx] = value[other_agent_idx]

        agent1_action = action[0].cpu().item()
        agent2_action = action[1].cpu().item()

        next_obs, reward, next_done, env_info = env.step((agent1_action, agent2_action))
        r1 = env_info['sparse_r_by_agent'][0] + env_info['shaped_r_by_agent'][0]
        r2 = env_info['sparse_r_by_agent'][1] + env_info['shaped_r_by_agent'][1]
        tot_rew += (r1 + r2)

        rewards[step, agent_idx] = (r1 + r2)
        rewards[step, other_agent_idx] = (r1 + r2)

        agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs[
            'other_agent_env_idx']
        next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)

    next_values = agent.get_value(next_obs).squeeze(1)
    return obs, actions, logprobs, values, next_values, rewards, dones, final_rew


def compute_loss(agent, b_obs, b_actions, b_logprobs, b_returns, b_advantages):
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions.long())
    logratio = newlogprob - b_logprobs
    ratio = logratio.exp()

    mb_advantages = b_advantages
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    # Not clip loss
    v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss


def project_conflicting_gradient(grad_i, grad_j):
    dot_product = torch.dot(grad_i, grad_j)
    if dot_product < 0:
        grad_j_norm = torch.norm(grad_j) ** 2 + 1e-10  # avoid div by 0
        projection = (dot_product / grad_j_norm) * grad_j
        return grad_i - projection
    else:
        return grad_i


def apply_pcgrad(gradient_list):
    pcgrads = []
    for i in range(len(gradient_list)):
        g_i = gradient_list[i].clone()
        for j in range(len(gradient_list)):
            if i != j:
                g_j = gradient_list[j]
                g_i = project_conflicting_gradient(g_i, g_j)
        pcgrads.append(g_i)

    #  Weight based on performance
    recent_avg_rewards = torch.tensor([np.mean(reward_history[l]) for l in layouts])
    weights = 1.0 / (recent_avg_rewards + 1e-6)  # Lower reward → higher weight
    weights = weights / weights.sum()

    weights = torch.tensor(weights, device=pcgrads[0].device, dtype=torch.float32)  # Shape: [num_tasks]
    stacked = torch.stack(pcgrads, dim=0)  # Shape: [num_tasks, grad_dim]

    # Weighted sum over task dimension
    final_grad = torch.sum(stacked * weights[:, None], dim=0)  # Shape: [grad_dim]
    return final_grad


if __name__ == "__main__":
    args = tyro.cli(Args)

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print(device)

    layouts = [
        "cramped_room",
        "asymmetric_advantages",
        "coordination_ring",
        "forced_coordination",
        "counter_circuit_o_1order",
    ]
    args.num_iterations = args.total_timesteps // (args.num_steps * len(layouts))

    env = GeneralizedOvercooked(layouts, horizon=args.horizon, use_r_shaped=True, old_dynamics=True)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = PPOActorCriticResidual(input_dim, output_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)

    global_step = 0
    reward_history = defaultdict(lambda: deque(maxlen=10))

    pbar = tqdm(range(args.num_iterations), desc="Training", dynamic_ncols=True)
    start_time = time.time()
    for i in pbar:
        grads_buff = []
        rewards_log = defaultdict(list)
        for layout_id in range(len(layouts)):
            obs, actions, logprobs, values, next_values, rewards, dones, final_rew = collect_rollouts(env, layout_id, agent)
            rewards_log[layouts[layout_id]].append(final_rew)
            reward_history[layouts[layout_id]].append(final_rew)

            advantages, returns = compute_multiagent_gae(rewards, values, dones, next_values, args.gamma, args.gae_lambda)

            b_obs = obs.reshape((-1,) + (input_dim,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            loss = compute_loss(agent, b_obs, b_actions, b_logprobs, b_returns, b_advantages)

            optimizer.zero_grad()
            loss.backward()

            # Flatten grads into a single vector
            param_with_grad = [p for p in agent.parameters() if p.requires_grad and p.grad is not None]
            flat_grad = torch.cat([p.grad.view(-1) for p in param_with_grad])

            grads_buff.append(flat_grad)

            global_step += args.num_steps

            # Apply PCGrad logic
        pcgrad = apply_pcgrad(grads_buff)

        # Assign projected gradients back to model
        offset = 0
        for p in agent.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.grad = pcgrad[offset:offset + numel].view_as(p).clone()
                offset += numel

        optimizer.step()

        # Logging reward stats in tqdm
        step_per_second = int(global_step // (time.time() - start_time))
        reward_str = f"{step_per_second} step/s,  " + " | ".join([f"{layout[:5]}: {np.mean(list(deque_vals)):.2f}" for layout, deque_vals in reward_history.items()])
        pbar.set_postfix_str(f"AvgRewards: {reward_str}")