import torch

from algorithms.ppo import PPOActorCritic


class PPOAgent:
    def __init__(self, model_path, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = PPOActorCritic(input_dim, output_dim)

        try:
            self.agent.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Model {self.agent.__class__.__name__} loaded correctly")
        except FileNotFoundError:
            print(f"Model {self.agent.__class__.__name__} to load not found")

    def action(self, state):
        with torch.no_grad():
            action, probs = self.agent.get_action_and_distribution(torch.tensor(state, dtype=torch.float32))
        return action.cpu().item(), probs.probs.exp().cpu().numpy()

    def actions(self, states):
        with torch.no_grad():
            action, probs = self.agent.get_action_and_distribution(torch.tensor(states))
        return action.cpu().numpy(), probs.probs.exp().cpu().numpy()