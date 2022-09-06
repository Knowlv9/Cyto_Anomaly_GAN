
import os, sys
from PIL import Image
from natsort import natsorted
from torch import nn, optim
from torch.utils.data import Dataset
# from torchsummary import summary
import matplotlib.pyplot as plt

class LoadFromFolder(Dataset):
	def __init__(self, imgs_list, main_dir, transform):
		self.main_dir = main_dir
		self.transform = transform
		all_imgs = natsorted(imgs_list)
		self.all_imgs_name = natsorted(all_imgs)
		self.imgs_loc = [os.path.join(self.main_dir, i) for i in self.all_imgs_name]

	def __len__(self):
		return len(self.all_imgs_name)

	def load_image(self, path):
		image = Image.open(path).convert("RGB")
		tensor_image = self.transform(image)
		return tensor_image

	def __getitem__(self, idx):

		# 後ほどsliceで画像を複数枚取得したいのでsliceでも取れるようにする
		if type(idx) == slice:
			paths = self.imgs_loc[idx]
			tensor_image = [self.load_image(path) for path in paths]
			tensor_image = torch.cat(tensor_image).reshape(len(tensor_image), *tensor_image[0].shape)
		elif type(idx) == int:
			path = self.imgs_loc[idx]
			tensor_image = self.load_image(path)
		return tensor_image
