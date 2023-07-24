import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

class Mnist_2NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, 10)

	def forward(self, inputs):
		tensor = F.relu(self.fc1(inputs))
		tensor = F.relu(self.fc2(tensor))
		tensor = self.fc3(tensor)
		return tensor


class Mnist_CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, 10)

	def forward(self, inputs):
		tensor = inputs.view(-1, 1, 28, 28)
		tensor = F.relu(self.conv1(tensor))
		tensor = self.pool1(tensor)
		tensor = F.relu(self.conv2(tensor))
		tensor = self.pool2(tensor)
		tensor = tensor.view(-1, 7*7*64)
		tensor = F.relu(self.fc1(tensor))
		tensor = self.fc2(tensor)
		return tensor

# Define a custom Dataset class
class UJIIndoorLocDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


## network structure ##
class FedCNN_BFC(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(520, 64)
		self.l2 = nn.ReLU()
		self.l3 = nn.Conv1d(1, 99, 22)
		self.l4 = nn.Conv1d(99, 66, 22)
		self.l5 = nn.Conv1d(66, 33, 22)
		self.l6 = nn.Linear(33, 8)
		self.l7 = nn.Sigmoid()

	def forward(self, x):
		x = self.l1(x)
		x = self.l2(x)
		x = x.reshape((-1, 1, 64))
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = x.reshape((-1, 1, 33))
		x = self.l6(x)
		x = self.l7(x)
		return x


class FedADA_LLR(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(520, 64)
		self.l2 = nn.ReLU()
		self.l3 = nn.Linear(64, 20)
		self.l4 = nn.ReLU()
		self.l5 = nn.Linear(20, 10)
		self.l6 = nn.ReLU()
		self.l7 = nn.Linear(10, 10)
		self.l8 = nn.ReLU()
		self.l9 = nn.Linear(10, 10)
		self.la = nn.ReLU()
		self.lb = nn.Linear(10, 2)

	def forward(self, x):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = self.l6(x)
		x = self.l7(x)
		x = self.l8(x)
		x = self.l9(x)
		x = self.la(x)
		x = self.lb(x)
		return x