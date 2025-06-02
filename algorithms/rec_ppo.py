from datetime import datetime
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
from dataclasses import dataclass
import tyro
import numpy as np
from tqdm import tqdm
import time
import random
import os

from utils.GeneralizedOvercooked import GeneralizedOvercooked
from utils.utils import count_params

os.environ["WANDB_SILENT"] = "True"
import wandb
load_dotenv()  # Loads .env into os.environ
wandb.login(key=os.getenv("WANDB_API_KEY"))
MODELS_FOLDER = 'saved_models'
os.makedirs(os.path.join(os.getcwd(), MODELS_FOLDER), exist_ok=True)

@dataclass
class Args:
    horizon: int = 400
    cuda: bool = True
    total_timesteps: int = int(5e5)
    learning_rate: float = 6e-4
    num_envs: int = 1
    num_agents: int = 2
    num_steps: int = 1200  # longer rollouts help reduce variance
    gamma: float = 0.9
    gae_lambda: float = 0.9
    update_epochs: int = 6  # more to stabilize updates
    clip_coef: float = 0.1
    val_clip_coef: float = 1
    ent_coef: float = 0.02  # encourages exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.1
    batch_size: int = num_steps
    minibatch_size: int = 80
    num_minibatches: int = batch_size // minibatch_size  # So when training we train for exactly one episode and than reset
    num_iterations: int = total_timesteps // batch_size


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RecurrentPPOActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size

        # Common feature extractor
        self.fc = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.ReLU(),
        )

        # Recurrent layer
        self.lstm = nn.LSTM(64, hidden_size)

        # Actor head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, output_dim), std=0.01),
        )

        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    @torch.no_grad
    def get_value(self, x, hxs):
        x = self.fc(x)
        x, hxs = self.lstm(x.unsqueeze(0), hxs)
        return self.critic(x.squeeze(0)), hxs

    def get_action_and_value(self, x, hxs, action=None):
        x = self.fc(x)
        x, hxs = self.lstm(x, hxs)
        x = x.squeeze(0)

        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        value = self.critic(x)
        return action, dist.log_prob(action), dist.entropy(), value, hxs

    def get_action_and_distribution(self, x, hx):
        x = self.fc(x)
        x, hx = self.lstm(x, hx)
        x = x.squeeze(0)

        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs, hx

    @torch.no_grad
    def init_hidden(self, batch_size, device="cpu"):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))


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
    device = "cuda"

    layouts = [
         "counter_circuit_o_1order",
         "cramped_room",
         "forced_coordination",
         "coordination_ring",
        "asymmetric_advantages",
    ]
    curriculum_goal = None

    set_seed(1)
    args = tyro.cli(Args)

    env = GeneralizedOvercooked(layouts, horizon=400, use_r_shaped=True, old_dynamics=True,
                                curriculum_goal=curriculum_goal)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = RecurrentPPOActorCritic(input_dim, output_dim).to(device)
    print("Total parameters:", count_params(agent))
    optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)

    #  Logging
    extra_info = ""
    wandb_run_name = f"PPO-LSTM_{', '.join(layouts)}_{datetime.now().strftime('%d-%m_%H-%M')}"
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
            "val_clip_coef": args.val_clip_coef,
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
    max_single_rew = 0

    agent_idx = 0
    other_agent_idx = 1
    start_time = time.time()
    for i in loop:
        next_obs, _ = env.reset()
        agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
        next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
        next_done = False
        hx = agent.init_hidden(args.num_agents, device)  # Assuming 1 LSTM layer and batch_size = num_agents
        for step in range(args.num_steps):
            obs[step, agent_idx] = next_obs[agent_idx]
            obs[step, other_agent_idx] = next_obs[other_agent_idx]

            with torch.no_grad():
                action, logprob, _, value, hx = agent.get_action_and_value(next_obs.unsqueeze(0), hx)

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
            tot_reward += (r1 + r2)

            rewards[step, agent_idx] = (r1 + r2)
            rewards[step, other_agent_idx] = (r1 + r2)
            dones[step, agent_idx] = next_done
            dones[step, other_agent_idx] = next_done

            agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
            next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)

            step_per_second = int(global_step // (time.time() - start_time))
            loop.set_postfix_str(
                f"{step_per_second} step/s, avg_reward={avg_rew:.1f}, last_rew={last_tot_rew}, max_rew={max_single_rew}, global_step={global_step}, loss={loss}")
            global_step += args.num_envs

            if next_done:
                hx = agent.init_hidden(args.num_agents, device)
                next_obs, is_new_layout = env.reset(avg_rew)
                agents_obs, env_state, _ = next_obs['both_agent_obs'], next_obs['overcooked_state'], next_obs['other_agent_env_idx']
                next_obs = torch.tensor(agents_obs, dtype=torch.float32).to(device)
                # Update reward tracking
                reward_buff.append(tot_reward)
                last_tot_rew = tot_reward
                max_single_rew = max(max_single_rew, tot_reward)
                tot_reward = 0
                # Curriculum trick: penalize layout switches with dummy rewards
                if is_new_layout:
                    reward_buff.extend([0] * 10)
                # Compute average recent reward
                avg_rew = np.mean(reward_buff[-10:])

        assert ((dones[:, 0] == 1) & (dones[:, 1] == 1)).sum().item() == (args.num_steps // args.horizon)
        #  Rollout finished, advantage calculation
        next_values, _ = agent.get_value(next_obs, hx)
        advantages, returns = compute_multiagent_gae(rewards, values, dones, next_values.squeeze(1), args.gamma, args.gae_lambda)

        b_obs = obs  # shape: [T, B, obs_dim]
        b_logprobs = logprobs  # [T, B]
        b_actions = actions  # [T, B]
        b_advantages = advantages  # [T, B]
        b_returns = returns  # [T, B]

        T, B = b_obs.shape[:2]
        batch_inds = np.arange(T)

        avg_grad_norm_buff = []
        clipfracs = []
        for epoch in range(args.update_epochs):

            for start in range(0, T, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = batch_inds[start:end]

                mb_obs = b_obs[mb_inds]  # [mb, B, obs_dim]
                mb_actions = b_actions[mb_inds]  # [mb, B]
                mb_logprobs = b_logprobs[mb_inds]  # [mb, B]
                mb_advantages = b_advantages[mb_inds]  # [mb, B]
                mb_returns = b_returns[mb_inds]  # [mb, B]

                if start % args.horizon == 0:
                    # Initialize LSTM hidden state, since each cycle math one episode
                    hxs = agent.init_hidden(B, device)
                    hxs = (hxs[0].detach(), hxs[1].detach())
                    assert list(dones[start:end][0]) == [0, 0]

                # Forward pass
                _, newlogprob, entropy, newvalue, hxs = agent.get_action_and_value(mb_obs, hxs, action=mb_actions)
                hxs = (hxs[0].detach(), hxs[1].detach())

                # Compute log ratio and ratio
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                # Normalize advantages per batch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                #v_loss = 0.5 * ((newvalue.squeeze(-1) - mb_returns) ** 2).mean()
                newvalue = newvalue.squeeze(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -args.val_clip_coef,
                    args.val_clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                losses.append(loss.item())

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

        # Logging
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

        # Optional model saving
        if i % 50 == 0:
            torch.save(agent.state_dict(), model_path)
