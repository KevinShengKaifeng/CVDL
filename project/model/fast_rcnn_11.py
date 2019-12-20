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
			nn.Linear(512 * roi_size * roi_size, 4096),
			nn.ReLU(True),
		)
		self.make_fc8 = nn.Sequential(
			nn.Linear(4096, 4096),
			nn.ReLU(True),
		)
		# use ModuleList
		self.Wk_list_1 = nn.ModuleList([nn.Linear(4096, 256) for i in range(16)])
		self.Wk_list_2 = nn.ModuleList([nn.Linear(4096, 256) for i in range(16)])
		self.Wq_list_1 = nn.ModuleList([nn.Linear(4096, 256) for i in range(16)])
		self.Wq_list_2 = nn.ModuleList([nn.Linear(4096, 256) for i in range(16)])
		self.Wv_list_1 = nn.ModuleList([nn.Linear(4096, 256) for i in range(16)])
		self.Wv_list_2 = nn.ModuleList([nn.Linear(4096, 256) for i in range(16)])
		self.classifier = nn.Linear(4096, 4096)
		self.cls_loc = nn.Linear(4096, n_class * 4)
		self.score = nn.Linear(4096, n_class)
		
		# initialize the weights
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

		# pooling to the same size
		# num_rois * (512 * 7 * 7)
		pool = self.roi(x, indices_and_rois)
		pool_size0 = pool.size(0)
		pool = pool.view(pool_size0, -1)
		
		maxx = t.zeros(pool_size0, pool_size0).cuda() + 1e-3
		
		fc7 = self.make_fc7(pool)
		
		# compute relation: wa and w without location info
		# first layer
		for i in range(16):
			Wk = self.Wk_list_1[i](fc7)
			Wq = self.Wq_list_1[i](fc7)
			Wv = self.Wv_list_1[i](fc7)
			pwa = Wk.mm(Wq.t()) / 16
			hwa, _ = t.max(pwa, 0)
			wa = (t.exp(pwa - hwa).view(pool_size0 * pool_size0, -1)).view(pool_size0, -1)
			sig_w = t.sum(wa)
			w = wa / sig_w
			if (i == 0):
				relation1 = w.t().mm(Wv)
			else:
				relation1 = t.cat((relation1, w.t().mm(Wv)), 1)

		fc8_input = fc7	+ relation1
		# second layer
		fc8 = self.make_fc8(fc8_input)

		for i in range(16):
			Wk = self.Wk_list_2[i](fc8)
			Wq = self.Wq_list_2[i](fc8)
			Wv = self.Wv_list_2[i](fc8)
			pwa = Wk.mm(Wq.t()) / 16
			hwa, _ = t.max(pwa, 0)
			wa = (t.exp(pwa - hwa).view(pool_size0 * pool_size0, -1)).view(pool_size0, -1)
			sig_w = t.sum(wa)
			w = wa / sig_w
			if (i == 0):
				relation2 = w.t().mm(Wv)
			else:
				relation2 = t.cat((relation2, w.t().mm(Wv)), 1)

		fc9_input = fc8 + relation2

		fc_cls = self.classifier(fc9_input) 
		
		roi_cls_locs = self.cls_loc(fc_cls)
		roi_scores = self.score(fc_cls)
		return roi_cls_locs, roi_scores
