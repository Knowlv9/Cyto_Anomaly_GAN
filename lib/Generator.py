from collections import OrderedDict
from torch import nn, optim
from torch.nn import functional as F
import torch
import sys
from torchvision import models
from torchinfo import summary

def convrelu(in_channels, out_channels, kernel, stride, padding):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.LeakyReLU(0.2, inplace=True),
	)

def conv_t_relu(in_channels, out_channels, kernel, stride, padding):
	return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.LeakyReLU(0.2, inplace=True),
	)

class Generator(nn.Module):
	def __init__(self, EMBED_SIZE):
		super().__init__()
		self.classname = "MildConvGenerator"

		self.main = nn.Sequential(
			conv_t_relu(EMBED_SIZE, 256, kernel=8, stride=1, padding=0),

			conv_t_relu(256, 128, kernel=4, stride=2, padding=1),

			conv_t_relu(128, 64, kernel=4, stride=2, padding=1),

			conv_t_relu(64, 32, kernel=4, stride=2, padding=1),

			conv_t_relu(32, 16, kernel=4, stride=2, padding=1),

			conv_t_relu(16, 3, kernel=4, stride=2, padding=1),

			nn.Tanh(),
		)

	def forward(self, z):
		out = self.main(z)
		return out
