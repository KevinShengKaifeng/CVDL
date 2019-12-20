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
		self.classifier = nn.Linear(4096, 4096)
		self.make_Wk = nn.Linear(4096, 4096)
		self.make_Wq = nn.Linear(4096, 4096)
		self.make_Wv = nn.Linear(4096, 4096)
		self.make_wg = nn.Sequential(
			nn.Linear(4, 1),
			nn.ReLU(True),
		)
		self.cls_loc = nn.Linear(4096, n_class * 4)
		self.score = nn.Linear(4096, n_class)

		self.make_wg[0].weight.data.normal_(10, 0.01)
		self.make_wg[0].bias.data.zero_()
		self.cls_loc.weight.data.normal_(0, 0.001)
		self.cls_loc.bias.data.zero_()
		self.score.weight.data.normal_(0, 0.01)
		self.score.bias.data.zero_()

		self.n_class = n_class
		self.roi_size = roi_size
		self.spatial_scale = spatial_scale
		self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

	def forward(self, x, rois, roi_indices):
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
		Wk = self.make_Wk(fc7)
		Wq = self.make_Wq(fc7)
		Wv = self.make_Wv(fc7)
		
		# compute wg
		x1 = 0.5 * (xy_indices_and_rois[:, 1] + xy_indices_and_rois[:, 3])
		y1 = 0.5 * (xy_indices_and_rois[:, 2] + xy_indices_and_rois[:, 4])
		w1 = t.abs(x1 - xy_indices_and_rois[:, 3])
		h1 = t.abs(y1 - xy_indices_and_rois[:, 4])
		x2 = x1.contiguous().view(-1, pool_size0).cuda()
		x1 = x2.view(pool_size0, -1).cuda()
		y2 = y1.contiguous().view(-1, pool_size0).cuda()
		y1 = y2.view(pool_size0, -1).cuda()
		w1 = w1.view(pool_size0, -1).cuda()
		h1 = h1.view(pool_size0, -1).cuda()
		
		tx = t.max(t.abs(x1 - x2), maxx) / w1
		ty = t.max(t.abs(y1 - y2), maxx) / h1
		tw = w1 / w1.view(-1, pool_size0).cuda()
		th = h1 / h1.view(-1, pool_size0).cuda()
#		same as but much faster
#		for i in range(32):
#			for j in range(32):
#				tmp[i, j] = tmp[i, j] / w1[i][0]
#		tmp = tmp / w1
#		print(t.broadcast_minus(x1, x1))
#		for i in range(pool_size0):
#			xm = x1[i][0]
#			ym = y1[i]
#			wm = w1[i, 0]
#			hm = h1[i]
#			for j in range(pool_size0):
#				if (i == j):
#					continue
#				xn = x1[j][0]
#				yn = y1[j]
#				wn = w1[j, 0]
#				hn = h1[j]
#				ps1[i, j] = abs(xm - xn) / wm
#				ps_one[i, j, 0] = abs(xm - xn) / wm
#				ps_one[i, j, 1] = abs(ym - yn) / hm
#				ps1[i, j] = wm / wn
#				ps_one[i, j, 3] = hn / hm
#		tg = t.tensor(pool_size0, pool_size0, 4).cuda()
		tg = t.stack((tx, ty, tw, th), dim = 2)
		ps_one = Variable(tg.view(pool_size0 * pool_size0, -1), requires_grad=False).cuda()
		wg = self.make_wg(ps_one) + 1
				
		# compute wa and w

#		same as but much faster
#		for i in range(pool_size0):
#			for j in range(pool_size0):
#				pwa[i, j] = pwa[i, j] + t.sum(Wk[i] * Wq[j])
		pwa = Wk.mm(Wq.t()) / 64
		
#		same as but much faster
#		for i in range(pool_size0):
#			hwa[i] = hwa[i] + t.max(pwa[:, i])
		hwa, _ = t.max(pwa, 0)
		
#		same as but much faster
#		for i in range(pool_size0):
#			for j in range(pool_size0):
#				wa[i, j] = wa[i, j] + t.exp(pwa[i, j] - hwa[j])
		wa = (t.exp(pwa - hwa).view(pool_size0 * pool_size0, -1) * wg).view(pool_size0, -1)
		
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

#		same as but much faster
#		for i in range(pool_size0):
#			for j in range(pool_size0):
#				relation[i] = relation[i] + w[j, i] * Wv[j]
		relation = w.t().mm(Wv)
		
		fc8_input = fc7	+ relation

		fc_cls = self.classifier(fc8_input) 
		
		roi_cls_locs = self.cls_loc(fc_cls)
		roi_scores = self.score(fc_cls)
		return roi_cls_locs, roi_scores
