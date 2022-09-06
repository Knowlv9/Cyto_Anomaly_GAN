import os, sys
import slideio
from PIL import Image
import matplotlib.pyplot as plt

# image root path
main_path = ""
# file: wsi(.svs, ...etc)
files = os.listdir(main_path)

save_dir = ""
split_save_dir = ""

size = 256

for idx, f in enumerate(files):
	filename = os.path.splitext(f)[0]
	save_path = "%s/%s" % (save_dir, filename)

	save_path = save_path.replace(',', '')
	split_path = save_path.split("#")
	dis_name = split_path[0]

	save_path = "%s_%s" % (dis_name, split_path[1].split("_")[1])
	os.makedirs(save_path, exist_ok=True)
	image_path = "%s/%s" % (main_path, f)

	slide = slideio.open_slide(image_path, "SVS")
	scene = slide.get_scene(0)
	image = scene.read_block((7000, 8000, 32500, 32500))
	# plt.imsave("F:/AnoGAN/data/LC_datasets/%s.png" % (filename.replace("#", '').replace(',', '')), image)

	y = 8000
	i = 0
	while y < 40500:
		x = 7000
		while x < 39500:
			print(x,y)
			trimmed_img = scene.read_block((x, y, size, size))
			no = filename.replace("#", '').replace(',', '').split("_")[1]
			plt.imsave("%s/%s_%s_%s_%s.jpeg" % (save_path, no, x, y, size), trimmed_img)
			x += size
		y += size
