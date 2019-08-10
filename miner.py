import torch
import random
import numpy as np
import torch.nn as nn
from dqn import DQN


class Miner(object):
    def __init__(self, shape, epsilon, memory_capacity, target_replace_iter, batch_size, gamma):
        self.shape = shape
        self.epsilon = epsilon
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        self.gamma = gamma

        if self.shape == 'easy':
            self.out_length = 8 * 8
        elif self.shape == 'middle':
            self.out_length = 16 * 16
        elif self.shape == 'hard':
            self.out_length = 16 * 30

        self.eval_net, self.target_net = DQN(self.out_length), DQN(self.out_length)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, self.out_length * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.FloatTensor(x)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        if random.random() < self.epsilon:
            actions = self.eval_net(x)
            action = torch.max(actions, 1)[1].data.numpy()[0]
        else:
            action = random.randint(1, self.out_length)
            action -= 1
        return action

    def store_transition(self, s, a, r, s_):
        s = s.reshape(-1)
        s_ = s_.reshape(-1)
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = random.sample(range(self.memory_capacity), self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.out_length]).reshape((self.batch_size, 1, 8, 8))
        b_a = torch.LongTensor(b_memory[:, self.out_length:self.out_length+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.out_length+1:self.out_length+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.out_length:]).reshape((self.batch_size, 1, 8, 8))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_params(self):
        torch.save(self.eval_net.state_dict(), 'eval.pth')

    def load_params(self, path):
        self.eval_net.load_state_dict(torch.load(path))

# miner = Miner('easy', 0.9, 2000, 100, 32, 0.9)
# x = torch.zeros((1, 8, 8))
# miner.choose_action(x)
# miner.store_transition(x, 1, 1, x)
# miner.learn()
