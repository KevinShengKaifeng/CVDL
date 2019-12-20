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
		self.make_Wk1 = nn.Linear(4096, 4096)
		self.make_Wq1 = nn.Linear(4096, 4096)
		self.make_Wv1 = nn.Linear(4096, 4096)
		self.make_Wk2 = nn.Linear(4096, 4096)
		self.make_Wq2 = nn.Linear(4096, 4096)
		self.make_Wv2 = nn.Linear(4096, 4096)
		self.make_wg1 = nn.Sequential(
			nn.Linear(4, 1),
			nn.ReLU(True),
		)
		self.make_wg2 = nn.Sequential(
			nn.Linear(4, 1),
			nn.ReLU(True),
		)
		self.cls_loc = nn.Linear(4096, n_class * 4)
		self.score = nn.Linear(4096, n_class)

		self.make_wg1[0].weight.data.normal_(10, 0.01)
		self.make_wg1[0].bias.data.zero_()
		self.make_wg2[0].weight.data.normal_(10, 0.01)
		self.make_wg2[0].bias.data.zero_()
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
		Wk1 = self.make_Wk1(fc7)
		Wq1 = self.make_Wq1(fc7)
		Wv1 = self.make_Wv1(fc7)
		
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
		
		tg = t.stack((tx, ty, tw, th), dim = 2)
		ps_one = Variable(tg.view(pool_size0 * pool_size0, -1), requires_grad=False).cuda()
		wg1 = self.make_wg1(ps_one) + 1
		wg2 = self.make_wg2(ps_one) + 1
		
		# compute wa and w
		# first layer
		pwa1 = Wk1.mm(Wq1.t()) / 64 
		hwa1, _ = t.max(pwa1, 0)
		
		wa1 = (t.exp(pwa1 - hwa1).view(pool_size0 * pool_size0, -1) * wg1).view(pool_size0, -1)
		
		sig_w1 = t.sum(wa1, 0)
		w1 = wa1 / sig_w1
		relation1 = w1.t().mm(Wv1)
		fc8_input = fc7	+ relation1
		
		# second layer
		fc8 = self.make_fc8(fc8_input)
		Wk2 = self.make_Wk2(fc8)
		Wq2 = self.make_Wq2(fc8)
		Wv2 = self.make_Wv2(fc8)
		
		pwa2 = Wk2.mm(Wq2.t()) / 64 
		hwa2, _ = t.max(pwa2, 0)
		wa2 = (t.exp(pwa2 - hwa2).view(pool_size0 * pool_size0, -1) * wg2).view(pool_size0, -1)
		sig_w2 = t.sum(wa2, 0)
		w2 = wa2 / sig_w2
		relation2 = w2.t().mm(Wv2)
		fc9_input = fc8	+ relation2
		
		fc_cls = self.classifier(fc9_input) 		
		
		roi_cls_locs = self.cls_loc(fc_cls)
		roi_scores = self.score(fc_cls)
		return roi_cls_locs, roi_scores
