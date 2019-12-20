from __future__ import division

import math
import torch as t
from torch import nn
from torchvision.models import vgg16
from utils import array_tool as at
from utils.config import opt
from torch.autograd import Variable
from model.roi_module import RoIPooling2D


class FastRCNN(nn.Module):
	def __init__(self, n_class = 21, roi_size = 7, spatial_scale = 1 / 16):
		# n_class includes the background
		super(FastRCNN, self).__init__()


		self.make_fc7 = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
		)
		self.classifier = nn.Linear(4096, 4096)
		self.make_Wk = nn.Linear(4096, 4096)
		self.make_Wq = nn.Linear(4096, 4096)
		self.make_Wv = nn.Linear(4096, 4096)
		self.cls_loc = nn.Linear(4096, n_class * 4)
		self.score = nn.Linear(4096, n_class)
		self.cls_loc.weight.data.normal_(0, 0.001)
		self.cls_loc.bias.data.zero_()
		self.score.weight.data.normal_(0, 0.01)
		self.score.bias.data.zero_()

		self.n_class = n_class
		self.roi_size = roi_size
		self.spatial_scale = spatial_scale
		self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

	def forward(self, x, rois, roi_indices):
		# in case roi_indices is  ndarray
		roi_indices = at.totensor(roi_indices).float()
		rois = at.totensor(rois).float()
		indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
		# NOTE: important: yx->xy
		xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
		indices_and_rois = Variable(xy_indices_and_rois.contiguous())

		pool = self.roi(x, indices_and_rois)
		pool_size0 = pool.size(0)
		pool = pool.view(pool_size0, -1)		
		ps_one = t.zeros(pool_size0, pool_size0, 4).cuda()
		
		fc7 = self.make_fc7(pool)
		Wk = self.make_Wk(fc7)
		Wq = self.make_Wq(fc7)
		Wv = self.make_Wv(fc7)

		# compute wa and w
#		same as but much faster
#		for i in range(pool_size0):
#			for j in range(pool_size0):
#				pwa[i, j] = pwa[i, j] + t.sum(Wk[i] * Wq[j])
		pwa = Wk.mm(Wq.t()) / 64
		
#		same as but much faster
#		print(pwa - pwa2)
#		for i in range(pool_size0):
#			hwa[i] = hwa[i] + t.max(pwa[:, i])
		hwa, _ = t.max(pwa, 0)
		
#		same as but much faster
#		for i in range(pool_size0):
#			for j in range(pool_size0):
#				wa[i, j] = wa[i, j] + t.exp(pwa[i, j] - hwa[j])
		wa = t.exp(pwa - hwa)
		
#		same as but much faster
#		for j in range(pool_size0):
#			for i in range(pool_size0):
#				sig_w[j] = sig_w[j] + wa[i, j]
		sig_w = t.sum(wa, 0)
		
#		same as but much faster
#		for j in range(pool_size0):
#			for i in range(pool_size0):
#				w[i, j] = wa[i, j] / sig_w[j]
		w = wa / sig_w

#		for i in range(pool_size0):
#			for j in range(pool_size0):
#				relation[i] = relation[i] + w[j, i] * Wv[j]
		relation = w.t().mm(Wv)
		fc8_input = fc7	+ relation

		fc_cls = self.classifier(fc8_input) 
		
		roi_cls_locs = self.cls_loc(fc_cls)
		roi_scores = self.score(fc_cls)
		return roi_cls_locs, roi_scores
