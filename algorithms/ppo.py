from datetime import datetime
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import wandb
from dataclasses import dataclass
import tyro
import numpy as np
from tqdm import tqdm
import time
import random
import os

from AutonomousSystemProject.GeneralizedOvercooked import GeneralizedOvercooked
from AutonomousSystemProject.utils import count_params

load_dotenv()  # Loads .env into os.environ
wandb.login(key=os.getenv("WANDB_API_KEY"))
MODELS_FOLDER = 'saved_models'
os.makedirs(os.path.join(os.getcwd(), MODELS_FOLDER), exist_ok=True)

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
    num_minibatches: int = 4  #
    update_epochs: int = 6  # more to stabilize updates
    clip_coef: float = 0.2
    ent_coef: float = 0.01  # encourages exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 100
    batch_size: int = num_steps * num_agents
    minibatch_size: int = batch_size // num_minibatches
    num_iterations: int = (total_timesteps * 2) // batch_size


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActorCritic(nn.Module):
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

    def get_action_and_distribution(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs


slice_intervals = [
    (0, 4),     # pi_orientation
    (4, 8),     # pi_obj

    (8, 10),   # pi_closest onion
    (10, 12),  # tomato
    (12, 14),  # dish
    (14, 16),  # soup
    (16, 18),  # serving
    (18, 20),  # counter

    (20, 22),   # pi_cloest_soup_n_{onions, tomatoes}

    # pi_closest_pot_0_* (10 features)
    (22, 23),  # pi_closest_pot_{j}_exists
    (23, 27),  # is_empty|is_full|is_cooking|is_ready
    (28, 30),  # num_onions|num_tomatoes
    (30, 31),  # cook_time
    (31, 33),  # pi_closest_pot_

    # pi_closest_pot_1_* (10 features)
    (33, 34),  # pi_closest_pot_{j}_exists
    (34, 41),  # is_empty|is_full|is_cooking|is_ready
    (41, 43),  # num_onions|num_tomatoes
    (43, 44),  # cook_time

    (44, 48),   # pi_wall (4)

    (48, 92),  # other player_features
    (92, 94),  # player_i_dist_to_other_players
    (94, 96)   # player_i-pos
]
class SelfAttentionActorCritic(nn.Module):
    def __init__(self, slice_intervals, output_dim):
        super(SelfAttentionActorCritic, self).__init__()
        self.slice_intervals = slice_intervals
        self.embed_dim = 16
        self.num_slices = len(slice_intervals)
        self.flat_dim = self.embed_dim * self.num_slices

        # Linear layers to project each slice to 16-dimensional embeddings
        self.slice_linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(end - start, self.embed_dim),
                nn.ReLU(),
            )
            for start, end in slice_intervals
        ])

        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=2, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.embed_dim)  # LayerNorm after attention

        # Final linear layers for actor and critic
        self.actor = nn.Sequential(
            nn.Linear(self.flat_dim, self.flat_dim),
            nn.ReLU(),
            nn.Linear(self.flat_dim, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.flat_dim, self.flat_dim),
            nn.ReLU(),
            nn.Linear(self.flat_dim, 1)
        )

    def get_attention_score(self, x):
        # x: (batch_size, 96)
        slices = []
        for (start, end), linear in zip(self.slice_intervals, self.slice_linears):
            sliced = x[:, start:end]  # (batch_size, slice_length)
            projected = linear(sliced)  # (batch_size, 16)
            slices.append(projected)

        # Stack slices to form (batch_size, num_slices, 16)
        tokens = torch.stack(slices, dim=1)

        # Self-attention expects (batch_size, seq_len, embed_dim)
        attended, _ = self.self_attention(tokens, tokens, tokens)  # (batch_size, num_slices, 16)

        # Flatten and apply final output layer
        attended_flat = attended.reshape(attended.size(0), -1)  # (batch_size, num_slices * 16)

        return attended_flat

    def get_value(self, x):
        att_score = self.get_attention_score(x)
        return self.critic(att_score)

    def get_action_and_value(self, x, action=None):
        att_score = self.get_attention_score(x)
        logits = self.actor(att_score)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(att_score)

    def get_action_and_distribution(self, x):
        att_score = self.get_attention_score(x)
        logits = self.actor(att_score)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_multiagent_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """
    rewards:     (T, N)
    values:      (T, N)
    dones:       (T, N)
    next_value:  (N,)
    """
    T, N = rewards.shape
    device = rewards.device

    adv = torch.zeros(T, N, device=device)
    last_gae_lam = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        adv[t] = last_gae_lam

    returns = adv + values
    return adv, returns


if __name__ == "__main__":
    args = tyro.cli(Args)
    set_seed(1)

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print(device)

    layouts = [
        #"counter_circuit_o_1order",
        "forced_coordination",
        #"coordination_ring",
        #"cramped_room",
        #"asymmetric_advantages",
    ]
    curriculum_goal = None
    env = GeneralizedOvercooked(layouts, horizon=args.horizon, use_r_shaped=True, old_dynamics=True, curriculum_goal=curriculum_goal)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = PPOActorCritic(input_dim, output_dim).to(device)
    print("Total parameters:", count_params(agent))
    optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)

    #  Logging
    extra_info = f""
    wandb_run_name = f"{agent.__class__.__name__}_{extra_info}_{', '.join(layouts)}_{datetime.now().strftime('%d-%m_%H-%M')}"
    if curriculum_goal:
        wandb_run_name += "_curriculum_levels"
    # Init wandb
    wandb.init(
        project="overcooked",
        name=wandb_run_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "clip_coef": args.clip_coef,
            "total_timesteps": args.total_timesteps,
            "layout": ", ".join(layouts),
            "episode_horizon": args.horizon,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "num_minibatches": args.num_minibatches,
            "update_epochs": args.update_epochs,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
        }
    )
    model_path = os.path.join(os.getcwd(), MODELS_FOLDER, f"{wandb_run_name}.pth")

    #  This is faster than using a RolloutBuffer since it stay in the GPU
    obs = torch.zeros((args.num_steps, args.num_agents) + (input_dim,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_agents)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_agents)).to(device)

    global_step = 0
    tot_reward = 0
    reward_buff = []
    loop = tqdm(range(args.num_iterations))
    avg_rew = 0
    loss = 0
    losses = []
    avg_grad_norm = 0
    last_tot_rew = 0

    next_obs, _ = env.reset()
    agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
    agent_idx = 0
    other_agent_idx = 1
    next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
    next_done = False
    start_time = time.time()
    for i in loop:

        for step in range(args.num_steps):
            if next_done:
                next_obs, is_new_layout = env.reset(avg_rew)
                agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
                next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
                # Update reward tracking
                reward_buff.append(tot_reward)
                last_tot_rew = tot_reward
                tot_reward = 0
                # Curriculum trick: penalize layout switches with dummy rewards
                if is_new_layout:
                    reward_buff.extend([0] * 10)
                # Compute average recent reward
                avg_rew = np.mean(reward_buff[-10:])

            global_step += args.num_envs

            obs[step, agent_idx] = next_obs[agent_idx]
            obs[step, other_agent_idx] = next_obs[other_agent_idx]

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
            step_rew = r1 + r2
            tot_reward += step_rew

            rewards[step, agent_idx] = step_rew
            rewards[step, other_agent_idx] = step_rew
            dones[step, agent_idx] = next_done
            dones[step, other_agent_idx] = next_done

            agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
            next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)

            step_per_second = int(global_step // (time.time() - start_time))
            loop.set_postfix_str(
                f"{step_per_second} step/s, avg_reward={avg_rew:.1f}, global_step={global_step}, loss={loss}")

        #  Rollout finished, advantage calculation
        next_values = agent.get_value(next_obs).squeeze(1)
        advantages, returns = compute_multiagent_gae(rewards, values, dones, next_values, args.gamma, args.gae_lambda)

        #  Net update
        b_obs = obs.reshape((-1,) + (input_dim, ))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        avg_grad_norm_buff = []
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                # Not clip loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                losses.append(loss.cpu().item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                # Compute average gradient norm
                total_norm = 0.0
                param_count = 0
                for p in agent.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                avg_grad_norm_buff.append((total_norm / param_count) ** 0.5)

                optimizer.step()

        wandb.log({
            "episode_reward": last_tot_rew,
            "avg_reward_10": avg_rew,
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "global_step": global_step,
            "clipfracs": np.mean(clipfracs),
            "average_grad_norm": np.mean(avg_grad_norm_buff),
            "max_avg_grad_norm": np.max(avg_grad_norm_buff),
            "old_kl_divergence": old_approx_kl.item(),
            "kl_divergence": approx_kl.item(),
            "entropy_loss": entropy_loss.item()
        })

        if i % 50 == 0:
            torch.save(agent.state_dict(), model_path)
