import os
import copy
import shutil
import datetime
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term, accuracy
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from timm.data import Mixup

from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
	from apex import amp
	from apex.parallel import DistributedDataParallel as ApexDDP
	from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
	from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN

from timm.utils import dispatch_clip_grad
from util.net import get_loss_scaler, get_autocast, distribute_bn

from ._base_trainer import BaseTrainer
from . import TRAINER


@TRAINER.register_module
class CLSTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(CLSTrainer, self).__init__(cfg)
		
	def set_input(self, inputs):
		self.imgs = inputs['img'].cuda()
		self.targets = inputs['target'].cuda()
		self.bs = self.imgs.shape[0]
	
	def forward(self, net=None):
		net = net if net is not None else self.net
		self.outputs = net(self.imgs)
		if not isinstance(self.outputs, dict):
			self.outputs = {'out': self.outputs, 'out_kd': self.outputs}
		
	def backward_term(self, loss_term, optim):
		optim.zero_grad()
		if self.loss_scaler:
			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
		else:
			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
			if self.cfg.loss.clip_grad is not None:
				dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
			optim.step()

	def optimize_parameters(self):
		if self.mixup_fn is not None:
			self.imgs, self.targets = self.mixup_fn(self.imgs, self.targets)
		with self.amp_autocast():
			self.forward()
			nan_or_inf_out = 1. if torch.any(torch.isnan(self.outputs['out'])) or torch.any(torch.isinf(self.outputs['out'])) else 0.
			nan_or_inf_out = reduce_tensor(nan_or_inf_out, self.world_size, mode='sum', sum_avg=False).clone().detach().item()
			nan_or_inf_out = True if nan_or_inf_out > 0. else False
			if nan_or_inf_out:
				self.nan_or_inf_cnt += 1
				log_msg(self.logger, f'NaN or Inf Found, total {self.nan_or_inf_cnt} times')
				self.check_bn()
			loss_ce = self.loss_terms['CE'](self.outputs['out'], self.targets) if not nan_or_inf_out else 0
			loss_kd = (self.loss_terms['KD'](self.outputs['out_kd'], self.imgs) if self.loss_terms.get('KD', None) else 0) if not nan_or_inf_out else 0
		self.backward_term((loss_ce + loss_kd) if not nan_or_inf_out else (0 * self.outputs['out'][0, 0]), self.optim)
		update_log_term(self.log_terms.get('CE'), reduce_tensor(loss_ce, self.world_size).clone().detach().item(), 1, self.master)
		update_log_term(self.log_terms.get('KD'), reduce_tensor(loss_kd, self.world_size).clone().detach().item(), 1, self.master)
		self._update_ema()

	@torch.no_grad()
	def test(self):
		tops = self.test_net(self.net, name='net')
		self.is_best = True if len(self.topk_recorder['net_top1']) == 0 or tops[0] > max(self.topk_recorder['net_top1']) else False
		self.topk_recorder['net_top1'].append(tops[0])
		self.topk_recorder['net_top5'].append(tops[1])
		max_top1 = max(self.topk_recorder['net_top1'])
		max_top1_idx = self.topk_recorder['net_top1'].index(max_top1) + 1
		msg = 'Max [top1: {:>3.3f} (epoch: {:d})]'.format(max_top1, max_top1_idx)
		if self.ema:
			tops = self.test_net(self.net_E, name='net_E')
			self.is_best_ema = True if len(self.topk_recorder['net_E_top1']) == 0 or tops[0] > max(
				self.topk_recorder['net_E_top1']) else False
			self.topk_recorder['net_E_top1'].append(tops[0])
			self.topk_recorder['net_E_top5'].append(tops[1])
			max_top1_ema = max(self.topk_recorder['net_E_top1'])
			max_top1_idx_ema = self.topk_recorder['net_E_top1'].index(max_top1_ema) + 1
			msg += ' [top1-ema: {:>3.3f} (epoch: {:d})]'.format(max_top1_ema, max_top1_idx_ema)
		log_msg(self.logger, msg)