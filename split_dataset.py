import random
import os, sys
import shutil
import datetime
import yaml
import pandas as pd
import numpy as np
import argparse
import datetime as dt

parser = argparse.ArgumentParser(description="AnoGAN interface script")
parser.add_argument('--path', type=str, default="E:/AnoGAN/datasets")
parser.add_argument("--anomaly_dirs", type=str, default="")
parser.add_argument("--frac", type=float, default="1.0")
parser.add_argument("--max_length", type=int, default="10000")
parser.add_argument("--val_length", type=int, default="100")

args = parser.parse_args()

def path_check(path):
	if os.path.exists(path) is False:
		print("wrong path: %s" %  path)
		sys.exit()

	return None

if __name__ == "__main__":
	#  train data directory path
	path_check(args.path)

	#  test anomaly directory path
	ano_path = args.anomaly_dirs
	path_check(ano_path)

	# list images in directory
	files = os.listdir(args.path)

	# count files
	n = int(len(files) * args.frac)
	n = args.max_length if n > args.max_length else n
	shuffle_files = random.sample(files, len(files))
	l = int(n if n > len(files) - n else len(files) - n)

	train_files = np.zeros([l]).astype(str)
	train_files[:] = ''
	train_files[:n] = shuffle_files[:n]

	val_len = args.val_length if n > args.val_length else n
	val_files = np.zeros([l]).astype(str)
	val_files[:] = ''
	val_files[:val_len] = random.sample(list(train_files[:n]), val_len)

	test_files = np.zeros([l]).astype(str)
	test_files[:] = ''

	test_files[:val_len] = shuffle_files[n:n+val_len]

	data = {
		"train": train_files,
		"val": val_files,
		"test": test_files
	}

	df = pd.DataFrame(data)

	today = datetime.datetime.now()
	today = today.strftime("%Y%m%d")

	dsts = os.listdir("./splits")
	i = len([f for f in dsts if today in f])
	file_path = f"./splits/{today}_{str(args.frac)}_{i}".replace(".", '') + ".csv"

	# save dataset
	print("save path: %s" % file_path)
	df.to_csv(file_path, encoding="utf-8")

	conf = {
		"exp_arguments": {
			"save_path": "LC_TEMP",
			"data_path": args.path,
			"datasets_csv_path": file_path,
		},
		"data_arguments": {
			"image_size": 256,
			"embed_size": 1024,
			"batch_size": 8,
			"epochs": 100,
			"lr": 0.0004
		},
		"default_setting": {
			"attach_path": "./results/AnoGAN",
			"anomaly_images": ano_path
		}
	}
	conf_path = f"./conf/config_{today}_{str(args.frac)}_{i}".replace(".", '') + ".yaml"
	print(conf_path)
	with open(conf_path, 'a', encoding="utf-8") as yf:
		yaml.dump(conf, yf)
