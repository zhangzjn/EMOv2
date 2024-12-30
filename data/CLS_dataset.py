import os
import glob
import json
import random
from torch.utils.data import dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS
from util.data import get_img_loader
from data.utils import get_transforms, get_scales
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torch.utils.data as data
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

from . import DATA


@DATA.register_module
class DefaultCLS(datasets.folder.DatasetFolder):  # ImageNet
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		root = '{}/{}'.format(cfg.data.root, 'train' if train else 'val')
		img_loader = get_img_loader(cfg.data.loader_type)
		super(DefaultCLS, self).__init__(root=root, loader=img_loader, extensions=IMG_EXTENSIONS, transform=transform, target_transform=target_transform)
		self.cfg = cfg
		self.train = train
		# scale_kwargs = cfg.trainer.scale_kwargs
		# if scale_kwargs is not None and scale_kwargs['n_scale'] > 0:
		# 	scale_kwargs = {k: v for k, v in scale_kwargs.items()}
		# 	self.scales = get_scales(**scale_kwargs)
		# else:
		# 	self.scales = [(cfg.size, cfg.size)]
		# self.num = 0
		# self.batch_size_per_gpu = cfg.trainer.data.batch_size_per_gpu

		self.nb_classes = cfg.data.nb_classes
		self.data_all = self.samples
		self.length = len(self.data_all)

	# def reset_scale_transform(self):
	# 	scale_rand = random.choices(self.scales, k=1)[0]
	# 	scale_rand = scale_rand[0]
	# 	self.cfg.size = scale_rand
	# 	self.cfg.data.train_transforms[0]['input_size'] = scale_rand
	# 	self.transform = get_transforms(self.cfg, train=True, cfg_transforms=self.cfg.data.train_transforms)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		# if len(self.scales) > 1 and self.num % self.batch_size_per_gpu == 0:
		# 	self.reset_scale_transform()
		# self.num += 1
		path, target = self.data_all[index]
		img = self.loader(path)
		img = self.transform(img) if self.transform is not None else img
		target = self.target_transform(target) if self.target_transform is not None else target

		return {'img': img, 'target': target}


class INatDataset(ImageFolder):
	def __init__(self, root, train=True, transform=None, year=2018):
		super(INatDataset, self).__init__(root=root)
		self.transform = transform
		self.year = year
		# assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
		category = 'name'
		path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
		with open(path_json) as json_file:
			data = json.load(json_file)
		with open(os.path.join(root, 'categories.json')) as json_file:
			data_catg = json.load(json_file)
		path_json_for_targeter = os.path.join(root, f"train{year}.json")
		with open(path_json_for_targeter) as json_file:
			data_for_targeter = json.load(json_file)
		targeter = {}
		indexer = 0
		for elem in data_for_targeter['annotations']:
			king = []
			king.append(data_catg[int(elem['category_id'])][category])
			if king[0] not in targeter.keys():
				targeter[king[0]] = indexer
				indexer += 1
		self.nb_classes = len(targeter)
		self.samples = []
		for elem in data['images']:
			cut = elem['file_name'].split('/')
			target_current = int(cut[2])
			path_current = os.path.join(root, cut[0], cut[2], cut[3])
			categors = data_catg[target_current]
			target_current_true = targeter[categors[category]]
			self.samples.append((path_current, target_current_true))

### ImageNet21K
# download link: https://opendatalab.com/ImageNet-21k/download
# 01: https://cdn.opendatalab.com/ImageNet-21k/raw/ImageNet21k-00.zip?Expires=1672760771&OSSAccessKeyId=LTAI5tCYLi1ZnJqYZX4tRk4q&Signature=N4NFPdRbLCQPYH6aT%2B9rISmeQ9Q%3D&response-content-type=application%2Foctet-stream
# 02: https://cdn.opendatalab.com/ImageNet-21k/raw/ImageNet21k-01.zip?Expires=1672760771&OSSAccessKeyId=LTAI5tCYLi1ZnJqYZX4tRk4q&Signature=07xGKO%2BN01MqrHnmJnrOlJwWrFU%3D&response-content-type=application%2Foctet-stream
# 03: https://cdn.opendatalab.com/ImageNet-21k/raw/ImageNet21k-02.zip?Expires=1672760771&OSSAccessKeyId=LTAI5tCYLi1ZnJqYZX4tRk4q&Signature=I6rWgueKX44byBdpvlne2YeZCgY%3D&response-content-type=application%2Foctet-stream
# 04: https://cdn.opendatalab.com/ImageNet-21k/raw/ImageNet21k-03.zip?Expires=1672760771&OSSAccessKeyId=LTAI5tCYLi1ZnJqYZX4tRk4q&Signature=uLMG9ndTodDAl81ltGNO73avRTM%3D&response-content-type=application%2Foctet-stream
# 05: https://cdn.opendatalab.com/ImageNet-21k/raw/ImageNet21k-04.zip?Expires=1672760771&OSSAccessKeyId=LTAI5tCYLi1ZnJqYZX4tRk4q&Signature=UCVQdB4Mei%2B2q1NUCe00rKwLKpM%3D&response-content-type=application%2Foctet-stream
# 06: https://cdn.opendatalab.com/ImageNet-21k/raw/ImageNet21k-05.zip?Expires=1672760771&OSSAccessKeyId=LTAI5tCYLi1ZnJqYZX4tRk4q&Signature=bdz69KI85tFHmmsQydK5JWR%2BhcM%3D&response-content-type=application%2Foctet-stream
# train-val split files: https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md
@DATA.register_module
class IN22KDataset(data.Dataset):
	def __init__(self, cfg, train=True, transform=None, target_transform=None):
		super(IN22KDataset, self).__init__()
		self.root = cfg.data.root
		self.loader = get_img_loader(cfg.data.loader_type)
		self.ann_path = f"{self.root}/ILSVRC2011fall_whole_map_{'train' if train else 'val'}.txt"
		self.cfg = cfg
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		# id & label: https://github.com/google-research/big_transfer/issues/7
		# total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
		self.nb_classes = cfg.data.nb_classes
		self.data_all = json.load(open(self.ann_path))
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		path, target = self.data_all[index]
		img = self.loader(f'{self.root}/{path}')
		img = self.transform(img) if self.transform is not None else img
		target = self.target_transform(target) if self.target_transform is not None else target

		return {'img': img, 'target': target}


@DATA.register_module
def Cifar10CLS(cfg, train=True, transforms=None):
	return datasets.CIFAR10(cfg.data.root, train=train, transform=transforms)


@DATA.register_module
def Cifar100CLS(cfg, train=True, transforms=None):
	return datasets.CIFAR100(cfg.data.root, train=train, transform=transforms)


@DATA.register_module
def INAT18CLS(cfg, train=True, transforms=None):
	return INatDataset(cfg.data.root, train=train, transforms=transforms, year=2018)


@DATA.register_module
def INAT19CLS(cfg, train=True, transforms=None):
	return INatDataset(cfg.data.root, train=train, transforms=transforms, year=2019)
