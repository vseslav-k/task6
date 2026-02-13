import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Transition:
    def __init__(self, obs, action, reward, resultingObs, done):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.resultingObs = resultingObs
        self.done = done


class QNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)



class DQNAgent:
    def __init__(self, env: gym.Env, learning_rate: float, init_exploration_rate: float, exploration_rate_decay: float, min_exploration_rate: float, future_reward_discount_factor: float = 0.95, 
                 retrain_frequency: int = 1, train_sample: int = 64, q_target_update_rate: int = 200, buffer_cap: int = 50000):
        
        self.env = env 
        self.learning_rate = learning_rate
        self.future_reward_discount_factor = future_reward_discount_factor
        self.exploration_rate = init_exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.min_exploration_rate = min_exploration_rate
        
        self.retrain_frequency = retrain_frequency
        self.train_sample = train_sample
        self.q_target_update_rate = q_target_update_rate

        self.transitions_storage = []
        self.buffer_cap = buffer_cap
        self.training_error = []
        self.update_counter = 0


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = self.env.action_space.n

        self.q_online_net = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.q_target_net = QNet(self.obs_dim, self.action_dim).to(self.device)

        self.sync_online_and_target()
        self.q_target_net.eval()

        self.optimizer = optim.Adam(self.q_online_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def recordTransition(self, obs, action, reward, resultingObs, done):
        if(len(self.transitions_storage) >= self.buffer_cap):
            self.transitions_storage.pop(0)
        self.transitions_storage.append(Transition(obs, action, reward, resultingObs, done))

    def obs_to_tensor(self, obs):
        return torch.tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)


    def getAction(self, obs=None):
        if (obs is None) or (random.random() < self.exploration_rate):
            return self.env.action_space.sample()

        with torch.no_grad():
            obs_t = self.obs_to_tensor(obs)
            q_values = self.q_online_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def Q_online(self, obs, action):
        # obs -> shape [1, obs_dim]
        obs_t = self.obs_to_tensor(obs)

        # forward pass -> shape [1, action_dim]
        q_values = self.q_online_net(obs_t)

        # select chosen action's Q value -> scalar tensor
        q_sa = q_values[0, int(action)]
        return q_sa
        
    def Q_target(self, obs, action):
        with torch.no_grad():
            obs_t = self.obs_to_tensor(obs)
            q_values = self.q_target_net(obs_t)
            q_sa = q_values[0, int(action)]
        return q_sa


    def sync_online_and_target(self):
        self.q_target_net.load_state_dict(self.q_online_net.state_dict())

    def calc_target(self, transition):
        # y = r                                 if done
        # y = r + gamma * max_a' Q_target(s',a') otherwise
        if transition.done:
            return torch.tensor(float(transition.reward), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_obs_t = self.obs_to_tensor(transition.resultingObs)      # [1, obs_dim]
            q_next_all = self.q_target_net(next_obs_t)                     # [1, action_dim]
            max_next_q = torch.max(q_next_all, dim=1).values[0]            # scalar
            y = float(transition.reward) + self.future_reward_discount_factor * max_next_q
            return y

    def calc_loss(self, batch):
        pred_q_list = []
        target_q_list = []

        for tr in batch:
            q_pred = self.Q_online(tr.obs, tr.action)   # scalar tensor
            y = self.calc_target(tr)                    # scalar tensor
            pred_q_list.append(q_pred)
            target_q_list.append(y)

        pred_q = torch.stack(pred_q_list)       # [B]
        y_q = torch.stack(target_q_list)        # [B]

        loss = self.loss_fn(pred_q, y_q)
        return loss
    
    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - self.exploration_rate_decay)

    def update_Q_online(self):
        if len(self.transitions_storage) < self.train_sample:
            return

        batch = random.sample(self.transitions_storage, self.train_sample)
        loss = self.calc_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_error.append(float(loss.item()))

    def update(self, obs, action, reward, resultingObs, done):
        if(obs is not None): 
            self.recordTransition(obs, action, reward, resultingObs, done)

        self.update_counter += 1

        if(self.update_counter % self.retrain_frequency == 0):
            self.update_Q_online()

        if(self.update_counter % self.q_target_update_rate == 0):
            self.sync_online_and_target()
        
    
train_episodes = 100

env = gym.make("CartPole-v1")
    
learning_rate = 1e-3
init_exploration_rate = 1
exploration_rate_decay = init_exploration_rate / train_episodes
min_exploration_rate = 0.07
agent = DQNAgent(env, learning_rate, init_exploration_rate, exploration_rate_decay, min_exploration_rate)
writer = SummaryWriter(log_dir="runs/cartpole_dqn")


def run_random_cartpole(episodes: int = 500, max_steps: int = 500, seed: int = 42, env_render_mode: str = None, training: bool = True):
    env = gym.make("CartPole-v1", render_mode=env_render_mode)
    agent.env = env

    global_step = 0
    recent_returns = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        ep_steps = 0

        for t in range(max_steps):
            action = agent.getAction(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if training:
                # store previous loss length so we can detect a new optimization step
                prev_loss_len = len(agent.training_error)

                agent.update(obs, action, reward, next_obs, truncated or terminated)

                # if a new optimization happened, log loss
                if len(agent.training_error) > prev_loss_len:
                    writer.add_scalar("train/loss", agent.training_error[-1], global_step)

            total_reward += reward
            obs = next_obs
            ep_steps += 1
            global_step += 1

            if terminated or truncated:
                break

        if training:
            agent.decay_exploration_rate()

        recent_returns.append(total_reward)
        if len(recent_returns) > 20:
            recent_returns.pop(0)
        avg20 = sum(recent_returns) / len(recent_returns)

        # Episode-level logs
        writer.add_scalar("episode/return", total_reward, ep)
        writer.add_scalar("episode/length", ep_steps, ep)
        writer.add_scalar("episode/epsilon", agent.exploration_rate, ep)
        writer.add_scalar("episode/return_avg20", avg20, ep)

        # Optional parameter norm (useful stability signal)
        with torch.no_grad():
            total_norm_sq = 0.0
            for p in agent.q_online_net.parameters():
                total_norm_sq += p.data.norm(2).item() ** 2
            param_l2 = total_norm_sq ** 0.5
        writer.add_scalar("model/q_online_param_l2", param_l2, ep)

        print(
            f"Episode {ep+1}: steps={ep_steps}, total_reward={total_reward:.1f}, "
            f"avg20={avg20:.1f}, eps={agent.exploration_rate:.3f}"
        )

    env.close()






# train
run_random_cartpole(episodes=train_episodes)


# test
run_random_cartpole(episodes = 100, env_render_mode = "human", seed = 43, training=False)

writer.close()