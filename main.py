import os, sys
import random
from glob import glob
import yaml, copy
import argparse
import logging.config
import datetime as dt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init as init
from torchvision.utils import make_grid, save_image

from torchinfo import summary

from lib.LoadFromFolder import LoadFromFolder

from lib.Encoder import Encoder
from lib.Generator import Generator
from lib.Discriminator import Discriminator

import warnings
warnings.simplefilter('ignore')

def Anomaly_score(x, E_x, G_E_x, Lambda=0.1):

	_,x_feature = model_D(x, E_x)
	_,G_E_x_feature = model_D(G_E_x, E_x)

	residual_loss = criterion_L1(x, G_E_x)
	discrimination_loss = criterion_L1(x_feature, G_E_x_feature)

	total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
	total_loss = total_loss.item()

	return total_loss

## init weight
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

## 時間測定開始
now = dt.datetime.now().strftime("%Y%m%d-%H%M")

parser = argparse.ArgumentParser(description="Efficient GAN interface script")
parser.add_argument('--config_file', type=str, default="./conf/config.yaml")
args = parser.parse_args()
conf = yaml.safe_load(open(args.config_file, 'r'))

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# image size
IMAGE_SIZE = int(conf["data_arguments"]["image_size"])
# demensions of latent variables
EMBED_SIZE = int(conf["data_arguments"]["embed_size"])
# batch size
BATCH_SIZE = conf["data_arguments"]["batch_size"]
EPOCHS = conf["data_arguments"]["epochs"]
# learning rate
LR = conf["data_arguments"]["lr"]

# current directory
ATTACH_PATH = conf["default_setting"]["attach_path"]

# save model
ATTACH_PATH = "%s/%s/%s_%s" % (ATTACH_PATH, conf["exp_arguments"]["save_path"], now, EPOCHS)
# save path for models
SAVE_MODEL_PATH = f"{ATTACH_PATH}/results/model/"
# save path for feature images
SAVE_IMAGE_FROM_Z_PATH = f"{ATTACH_PATH}/results/image_from_z/"
# save path for reconstruct images
SAVE_IMAGE_RECONSTRUCT = f"{ATTACH_PATH}/results/RECONSTRUCT/"

# summary(
# 	Encoder(EMBED_SIZE=EMBED_SIZE).to(device),
# 	tuple([1, 3, IMAGE_SIZE, IMAGE_SIZE]),
# 	depth=3,
# 	col_width = 15,
# 	col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
# )

# summary(
# 	Generator(EMBED_SIZE=EMBED_SIZE).to(device),
# 	tuple([1, EMBED_SIZE, 1, 1]),
# 	depth=3,
# 	col_width = 15,
# 	col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
# )

# summary(
# 	Discriminator(EMBED_SIZE=EMBED_SIZE).to(device),
# 	[(1, 3, IMAGE_SIZE, IMAGE_SIZE), (1, EMBED_SIZE, 1, 1)],
# 	depth=3,
# 	col_width = 15,
# 	col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
# )

sys.exit()


# create directories
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_FROM_Z_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_RECONSTRUCT, exist_ok=True)
os.makedirs("%s/train" % ATTACH_PATH, exist_ok=True)

alpha = 0.01

if __name__ == "__main__":

	#  path of images dataset
	data_path = conf["exp_arguments"]["data_path"]

	# path of csv splited train and test in dataset images
	datasets_csv_path = conf["exp_arguments"]["datasets_csv_path"]

	train_test_df = pd.read_csv(datasets_csv_path)

	train_imgs = list(train_test_df["train"])
	val_imgs = list(train_test_df["val"].dropna())
	test_imgs = list(train_test_df["test"].dropna())

	transform_dict = {
	"train": transforms.Compose(
			[
				transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
			]
		),
			"test": transforms.Compose(
			[
				transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
				transforms.ToTensor(),
			]
		)
	}

	train_dataset = LoadFromFolder(
		imgs_list=train_imgs,
		main_dir=data_path,
		transform=transform_dict["train"]
	)
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		**kwargs
	)

	test_dataset = LoadFromFolder(
		imgs_list = test_imgs,
		main_dir=data_path,
		transform=transform_dict["test"]
	)
	val_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		**kwargs
	)

	## define model
	model_E = Encoder(EMBED_SIZE=EMBED_SIZE).to(device)
	model_E.apply(weights_init)

	model_G = Generator(EMBED_SIZE=EMBED_SIZE).to(device)
	model_G.apply(weights_init)

	model_D = Discriminator(EMBED_SIZE=EMBED_SIZE).to(device)
	model_D.apply(weights_init)

	# Loss function: Binary Cross Entropy Loss
	criterion = nn.BCELoss()
	# calucurate for anomaly score
	criterion_L1 = nn.L1Loss(reduction="sum")

	# optimizer Encoder, Generator
	optimizer_ge =  torch.optim.Adam(
		list(model_G.parameters()) + list(model_E.parameters()),
		lr=LR/4, betas=(0.5,0.999)
	)
	# optimizer: Discriminator
	optimizer_d = torch.optim.Adam(model_D.parameters(), lr=LR, betas=(0.5,0.999))

	# lr decay： lr*0.9/50epoch
	scheduler_ge = torch.optim.lr_scheduler.StepLR(optimizer_ge, step_size=50, gamma=0.99)
	scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.99)

	loss_d_list, loss_ge_list, anomaly_score_list, epoch_list = [], [], [], []

	for epoch in range(int(EPOCHS)):
		loss_d_sum = 0
		loss_ge_sum = 0
		anomaly_score_sum = 0

		for i,(x, x_val) in enumerate(zip(train_loader, val_loader)):

			model_G.train()
			model_D.train()
			model_E.train()

			# set values
			y_true = Variable(torch.ones(x.size()[0])).to(device) - 0.1
			y_fake = Variable(torch.zeros(x.size()[0])).to(device)

			x = Variable(x).to(device)
			z = Variable(init.normal(torch.Tensor(x.size()[0],EMBED_SIZE, 1, 1),mean=0,std=0.1)).to(device)

			# noise for discriminator
			noise1 = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * (EPOCHS - epoch) / EPOCHS),
			requires_grad=False).to(device)
			noise2 = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * (EPOCHS - epoch) / EPOCHS),
			requires_grad=False).to(device)

			# discriminator
			optimizer_d.zero_grad()

			E_x = model_E(x)
			p_true, _ = model_D(x + noise1, E_x)

			G_z = model_G(z)
			p_fake, _ = model_D(G_z + noise2, z)


			loss_d = (criterion(p_true, y_true) + criterion(p_fake, y_fake))
			loss_d.backward(retain_graph=True)

			optimizer_d.step()

			# generator and encoder
			optimizer_ge.zero_grad()

			G_E_x = model_G(E_x)
			E_G_z = model_E(G_z)

			p_true, _ = model_D(x + noise1, E_x)

			# G_z = model_G(z)
			p_fake, _ = model_D(G_z + noise2, z)

			loss_ge_1 = criterion(p_fake, y_true) + criterion(p_true, y_fake)
			loss_ge_2 = criterion_L1(x, G_E_x) +  criterion_L1(z, E_G_z)

			loss_ge = (1 - alpha)*loss_ge_1 + alpha*loss_ge_2
			loss_ge.backward(retain_graph=True)
			optimizer_ge.step()

			loss_d_sum += loss_d.item()
			loss_ge_sum += loss_ge.item()

			# record anomaly score
			# evaluation
			model_G.eval()
			model_D.eval()
			model_E.eval()
			x_val = Variable(x_val).to(device)
			E_x_val = model_E(x_val)
			G_E_x_val = model_G(E_x_val)
			anomaly_score_sum += Anomaly_score(x_val, E_x_val, G_E_x_val)

			# save images
			if i == 0 and (epoch + 1) % 5 == 0:
				model_G.eval()
				model_D.eval()
				model_E.eval()

				save_image_size_for_z = min(BATCH_SIZE, 8)
				save_images = model_G(z)
				save_image(save_images[:save_image_size_for_z], f"{SAVE_IMAGE_FROM_Z_PATH}/epoch_{epoch+1}.png", nrow=4)

				save_image_size_for_recon = min(BATCH_SIZE, 8)
				images = x[:save_image_size_for_recon]
				G_E_x = model_G(model_E(images))
				diff_images = torch.abs(images - G_E_x)
				comparison = torch.cat([images , G_E_x, diff_images]).to("cpu")
				save_image(comparison, f"{SAVE_IMAGE_RECONSTRUCT}/epoch_{epoch+1}.png", nrow=save_image_size_for_recon)

			# lr decay
			scheduler_ge.step()
			scheduler_d.step()

			# record loss
			loss_d_mean = loss_d_sum / len(train_loader)
			loss_ge_mean = loss_ge_sum / len(train_loader)
			anomaly_score_mean = anomaly_score_sum / len(train_loader)

			print(f"{epoch+1}/{EPOCHS} epoch ge_loss: {loss_ge_mean:.3f} d_loss: {loss_d_mean:.3f} anomaly_score: {anomaly_score_mean:.3f}")

			epoch_list.append(epoch)
			loss_d_list.append(loss_d_mean)
			loss_ge_list.append(loss_ge_mean)
			anomaly_score_list.append(anomaly_score_mean)

			# save model
			if (epoch + 1) % 10 == 0:
				torch.save(model_G.state_dict(),f'{SAVE_MODEL_PATH}/Generator_{epoch + 1}.pkl')
				torch.save(model_E.state_dict(),f'{SAVE_MODEL_PATH}/Encoder_{epoch + 1}.pkl')
				torch.save(model_D.state_dict(),f'{SAVE_MODEL_PATH}/Discriminator_{epoch + 1}.pkl')

				loss_list = pd.DataFrame({
					"epoch": epoch_list,
					"Discriminator": loss_d_list,
					"Generator": loss_ge_list,
					"anomaly score": anomaly_score_list
				})
				loss_list.to_csv("%s/train/%sep_loss_data.csv" % (ATTACH_PATH, epoch+1))

	loss_list = pd.DataFrame({
		"Discriminator": loss_d_list,
		"Generator": loss_ge_list,
		"anomaly score": anomaly_score_list
	})
	save_path = "%s/train/total_loss_data.csv" % ATTACH_PATH
	print("loss eval data:", save_path)
	loss_list.to_csv(save_path, encoding="utf-8")
