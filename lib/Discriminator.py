from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import torch
import sys
from torchinfo import summary
from torchvision import models
from torch.autograd import Variable

def convrelu(in_channels, out_channels, kernel, stride, padding):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
		nn.BatchNorm2d(out_channels),
		nn.LeakyReLU(inplace=True),
		nn.Dropout2d(p=0.2)
	)

class Discriminator(nn.Module):
	def __init__(self, EMBED_SIZE, in_channels=3, out_channels=1, init_features=16):
		super(Discriminator, self).__init__()
		self.classname = "MildConvDiscriminator"
		features = init_features

		self.x_layer = nn.Sequential(
			convrelu(3, 16, 4, 2, 1),

			convrelu(16, 32, 4, 2, 1),
			
			convrelu(32, 64, 4, 2, 1),

			convrelu(64, 256, 4, 2, 1),

			convrelu(256, 1024, 4, 2, 1),

			convrelu(1024, 1024, 8, 1, 0),
		)
		self.z_layer = nn.Sequential(
			nn.Conv2d(EMBED_SIZE, 1024, kernel_size=1, stride=1),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Dropout2d(p=0.2),
		)

		self.last1 = nn.Sequential(
			nn.Conv2d(2048, 1024, kernel_size=1, stride=1),
			nn.LeakyReLU(0.1, inplace=False),
			nn.Dropout2d(p=0.2),
		)
		self.last2 = nn.Sequential(
			nn.Conv2d(1024, 1, kernel_size=1, stride=1),
		)

	def forward(self, x, z):
		output_x = self.x_layer(x) # x: 3, 256, 256

		output_z = self.z_layer(z) # z: 1024, 16, 16
		# return output_x
		concat_x_z = torch.cat((output_x, output_z), 1)

		output = self.last1(concat_x_z)
		feature = output.view(output.size()[0], -1)

		output = self.last2(output)
		output = F.sigmoid(output)
		return output.squeeze(), feature








	# @staticmethod
	# def _block(in_channels, features, name, kernel, stride, padding):
	# 	return nn.Sequential(
	# 	OrderedDict(
	# 	[
	# 	(
	# 	name + "conv1",
	# 	nn.Conv2d(
	# 	in_channels=in_channels,
	# 	out_channels=features,
	# 	kernel_size=kernel,
	# 	stride=stride,
	# 	padding=padding,
	# 	bias=False,
	# 	),
	# 	),
	# 	(name + "norm1", nn.BatchNorm2d(num_features=features)),
	# 	(name + "relu1", nn.LeakyReLU(inplace=True)),
	# 	(
	# 	name + "conv2",
	# 	nn.Conv2d(
	# 	in_channels=features,
	# 	out_channels=features,
	# 	kernel_size=3,
	# 	padding=1,
	# 	bias=False,
	# 	),
	# 	),
	# 	(name + "norm2", nn.BatchNorm2d(num_features=features)),
	# 	(name + "relu2", nn.LeakyReLU(inplace=True)),
	# 	]
	# 	)
	# 	)
