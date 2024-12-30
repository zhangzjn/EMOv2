from argparse import Namespace as _Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN as _IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD as _IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as _F
from data import get_dataset
from model import get_model
from optim import get_optim

# ---> test dataset
size = 224
data = _Namespace()
data.type = 'ImageFolderLMDB'
data.root = '/youtu_fuxi_team1_ceph/vtzhang/codes/data/imagenet'
data.loader_type = 'pil'
data.sampler = 'naive'
data.nb_classes = 1000

data.train_transforms = [
	dict(type='timm_create_transform', input_size=size, is_training=True, color_jitter=0.4,
		 auto_augment='rand-m9-mstd0.5-inc1', interpolation='random', mean=_IMAGENET_DEFAULT_MEAN, std=_IMAGENET_DEFAULT_STD,
		 re_prob=0.25, re_mode='pixel', re_count=1),
]
data.test_transforms = [
	dict(type='Resize', size=int(size / 0.875), interpolation=_F.InterpolationMode.BICUBIC),
	dict(type='CenterCrop', size=size),
	dict(type='ToTensor'),
	dict(type='Normalize', mean=_IMAGENET_DEFAULT_MEAN, std=_IMAGENET_DEFAULT_STD, inplace=True),
]

cfg = _Namespace()
cfg.data = data
train_dataset, test_dataset = get_dataset(cfg)
