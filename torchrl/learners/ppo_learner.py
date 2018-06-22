import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from torchrl.learners import BaseLearner
from torchrl.models import ActorCriticNet


class BasePPOLearner(BaseLearner):
  def __init__(self, observation_space, action_space,
               lr=1e-3,
               gamma=0.99,
               lmbda=0.01,
               alpha=0.5,
               beta=1.0,
               clip_ratio=0.2,
               max_grad_norm=1.0):
    super(BasePPOLearner, self).__init__(observation_space, action_space)

    self.ac_net = ActorCriticNet(observation_space.shape[0],
                                 action_space.shape[0], 256)
    self.ac_net_optim = Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta
    self.clip_ratio = clip_ratio
    self.max_grad_norm = max_grad_norm

  @property
  def models(self):
    return [self.ac_net]

  def act(self, obs):
    _, dist = self.ac_net(obs)
    action = dist.sample()
    return action.unsqueeze(1).cpu().numpy()

  def compute_returns(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    with torch.no_grad():
      values, dist = self.ac_net(obs)
      if not done[-1]:
        next_value, _ = self.ac_net(next_obs[-1:])
        values = torch.cat([values, next_value], dim=0)
      else:
        values = torch.cat([values, torch.zeros(1, 1)], dim=0)

      returns = torch.zeros(len(reward), 1)
      gae = 0.0
      for step in reversed(range(len(reward))):
        delta = reward[step] + self.gamma * values[step + 1] - values[step]
        gae = delta + self.gamma * self.lmbda * gae
        returns[step] = gae + values[step]

      log_probs = dist.log_prob(action).detach()
      values = values[:-1]  # remove the added step to compute returns

      return returns, log_probs, values

  def learn(self, obs, action, reward, next_obs, done,  #pylint: disable=unused-argument
            returns, old_log_probs, advantages):
    values, dist = self.ac_net(obs)

    new_log_probs = dist.log_prob(action)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                        1 + self.clip_ratio) * advantages
    actor_loss = - torch.min(surr1, surr2).mean()

    critic_loss = F.mse_loss(values, returns)

    entropy_loss = dist.entropy().mean()

    loss = actor_loss + self.alpha * critic_loss - self.beta * entropy_loss

    self.ac_net_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.ac_net.parameters(),
                                    self.max_grad_norm)
    self.ac_net_optim.step()

    return actor_loss.detach().cpu().item(), \
        critic_loss.detach().cpu().item(), \
        entropy_loss.detach().cpu().item()

  def save(self, save_dir):
    model_file_name = os.path.join(save_dir, 'ac_net.pth')
    torch.save(self.ac_net.state_dict(), model_file_name)

  def load(self, load_dir):
    model_file_name = os.path.join(load_dir, 'ac_net.pth')
    self.ac_net.load_state_dict(torch.load(model_file_name))
