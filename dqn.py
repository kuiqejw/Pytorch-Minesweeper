import torch.nn as nn
import torch.nn.init as init


class DQN(nn.Module):
	def __init__(self, out_length):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(1, 9, 3, 1, 0)
		# self.conv2 = nn.Conv2d(out_length, out_length, 3, 1, 0)
		# self.linear = nn.Linear(out_length * 16, out_length * 4)
		self.out = nn.Linear(324, out_length)
		self.relu = nn.PReLU()
		# init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
		# init.constant_(self.conv1.bias, 0.0)
		# init.kaiming_normal_(self.out.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
		# init.constant_(self.out.bias, 0.0)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		# print('First Step', x.shape)
		# x = self.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		# print('expand', x.shape)
		# x = self.relu(self.linear(x))
		out = self.out(x)
		# print(out.shape)
		return out