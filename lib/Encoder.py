## エンコーダー
from collections import OrderedDict
from torch import nn, optim
from torchvision import models
from torchinfo import summary
import numpy as np

def convrelu(in_channels, out_channels, kernel, stride, padding):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.LeakyReLU(inplace=True),
	)

class Encoder(nn.Module):
	def __init__(self, EMBED_SIZE, out_channels=2, in_channels=3):
		super(Encoder, self).__init__()
		self.classname = "MildConvEncoder"

		self.main = nn.Sequential(
			convrelu(3, 16, 4, 2, 1),

			convrelu(16, 32, 4, 2, 1),

			convrelu(32, 64, 4, 2, 1),

			convrelu(64, 256, 4, 2, 1),

			convrelu(256, 1024, 4, 2, 1),
			# 32×128×128
		)

		self.last = nn.Sequential(
			nn.Conv2d(1024, EMBED_SIZE, kernel_size=8, stride=1, bias=False)
		)

	def forward(self, x):
		out = self.main(x)

		out = self.last(out)
		out = out.view(out.size()[0], -1, 1, 1)
		return out
