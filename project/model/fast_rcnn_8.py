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
		self.make_wg1 = nn.Sequential(
			nn.Linear(4, 1),
			nn.ReLU(True),
		)
		self.make_wg2 = nn.Sequential(
			nn.Linear(4, 1),
			nn.ReLU(True),
		)
		self.classifier = nn.Linear(4096, 4096)
		self.make_Wk1_1 = nn.Linear(4096, 1024)
		self.make_Wq1_1 = nn.Linear(4096, 1024)
		self.make_Wv1_1 = nn.Linear(4096, 1024)
		self.make_Wk1_2 = nn.Linear(4096, 1024)
		self.make_Wq1_2 = nn.Linear(4096, 1024)
		self.make_Wv1_2 = nn.Linear(4096, 1024)
		self.make_Wk1_3 = nn.Linear(4096, 1024)
		self.make_Wq1_3 = nn.Linear(4096, 1024)
		self.make_Wv1_3 = nn.Linear(4096, 1024)
		self.make_Wk1_4 = nn.Linear(4096, 1024)
		self.make_Wq1_4 = nn.Linear(4096, 1024)
		self.make_Wv1_4 = nn.Linear(4096, 1024)
		self.make_Wk2_1 = nn.Linear(4096, 1024)
		self.make_Wq2_1 = nn.Linear(4096, 1024)
		self.make_Wv2_1 = nn.Linear(4096, 1024)
		self.make_Wk2_2 = nn.Linear(4096, 1024)
		self.make_Wq2_2 = nn.Linear(4096, 1024)
		self.make_Wv2_2 = nn.Linear(4096, 1024)
		self.make_Wk2_3 = nn.Linear(4096, 1024)
		self.make_Wq2_3 = nn.Linear(4096, 1024)
		self.make_Wv2_3 = nn.Linear(4096, 1024)
		self.make_Wk2_4 = nn.Linear(4096, 1024)
		self.make_Wq2_4 = nn.Linear(4096, 1024)
		self.make_Wv2_4 = nn.Linear(4096, 1024)
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
		
		# compute relation: wa and w
		# first layer
		fc7 = self.make_fc7(pool)
		Wk1_1 = self.make_Wk1_1(fc7)
		Wq1_1 = self.make_Wq1_1(fc7)
		Wv1_1 = self.make_Wv1_1(fc7)
		Wk1_2 = self.make_Wk1_2(fc7)
		Wq1_2 = self.make_Wq1_2(fc7)
		Wv1_2 = self.make_Wv1_2(fc7)
		Wk1_3 = self.make_Wk1_3(fc7)
		Wq1_3 = self.make_Wq1_3(fc7)
		Wv1_3 = self.make_Wv1_3(fc7)
		Wk1_4 = self.make_Wk1_4(fc7)
		Wq1_4 = self.make_Wq1_4(fc7)
		Wv1_4 = self.make_Wv1_4(fc7)
		
		pwa1_1 = Wk1_1.mm(Wq1_1.t()) / 32 
		hwa1_1, _ = t.max(pwa1_1, 0)
		wa1_1 = (t.exp(pwa1_1 - hwa1_1).view(pool_size0 * pool_size0, -1) * wg1).view(pool_size0, -1)
		pwa1_2 = Wk1_2.mm(Wq1_2.t()) / 32 
		hwa1_2, _ = t.max(pwa1_2, 0)
		wa1_2 = (t.exp(pwa1_2 - hwa1_2).view(pool_size0 * pool_size0, -1) * wg1).view(pool_size0, -1)
		pwa1_3 = Wk1_3.mm(Wq1_3.t()) / 32 
		hwa1_3, _ = t.max(pwa1_3, 0)
		wa1_3 = (t.exp(pwa1_3 - hwa1_3).view(pool_size0 * pool_size0, -1) * wg1).view(pool_size0, -1)
		pwa1_4 = Wk1_4.mm(Wq1_4.t()) / 32 
		hwa1_4, _ = t.max(pwa1_4, 0)
		wa1_4 = (t.exp(pwa1_4 - hwa1_4).view(pool_size0 * pool_size0, -1) * wg1).view(pool_size0, -1)
		
		sig_w1_1 = t.sum(wa1_1, 0)
		w1_1 = wa1_1 / sig_w1_1
		sig_w1_2 = t.sum(wa1_2, 0)
		w1_2 = wa1_2 / sig_w1_2
		sig_w1_3 = t.sum(wa1_3, 0)
		w1_3 = wa1_3 / sig_w1_3
		sig_w1_4 = t.sum(wa1_4, 0)
		w1_4 = wa1_4 / sig_w1_4
		
		# concat the blocks
		relation1 = t.cat((t.cat((w1_1.t().mm(Wv1_1), w1_2.t().mm(Wv1_2)), 1), t.cat((w1_3.t().mm(Wv1_3), w1_4.t().mm(Wv1_4)), 1)), 1)
		fc8_input = fc7	+ relation1

		# second layer
		fc8 = self.make_fc8(fc8_input)
		Wk2_1 = self.make_Wk2_1(fc8)
		Wq2_1 = self.make_Wq2_1(fc8)
		Wv2_1 = self.make_Wv2_1(fc8)
		Wk2_2 = self.make_Wk2_2(fc8)
		Wq2_2 = self.make_Wq2_2(fc8)
		Wv2_2 = self.make_Wv2_2(fc8)
		Wk2_3 = self.make_Wk2_3(fc8)
		Wq2_3 = self.make_Wq2_3(fc8)
		Wv2_3 = self.make_Wv2_3(fc8)
		Wk2_4 = self.make_Wk2_4(fc8)
		Wq2_4 = self.make_Wq2_4(fc8)
		Wv2_4 = self.make_Wv2_4(fc8)
		
		pwa2_1 = Wk2_1.mm(Wq2_1.t()) / 32 
		hwa2_1, _ = t.max(pwa2_1, 0)
		wa2_1 = (t.exp(pwa2_1 - hwa2_1).view(pool_size0 * pool_size0, -1) * wg2).view(pool_size0, -1)
		pwa2_2 = Wk2_2.mm(Wq2_2.t()) / 32 
		hwa2_2, _ = t.max(pwa2_2, 0)
		wa2_2 = (t.exp(pwa2_2 - hwa2_2).view(pool_size0 * pool_size0, -1) * wg2).view(pool_size0, -1)
		pwa2_3 = Wk2_3.mm(Wq2_3.t()) / 32 
		hwa2_3, _ = t.max(pwa2_3, 0)
		wa2_3 = (t.exp(pwa2_3 - hwa2_3).view(pool_size0 * pool_size0, -1) * wg2).view(pool_size0, -1)
		pwa2_4 = Wk2_4.mm(Wq2_4.t()) / 32 
		hwa2_4, _ = t.max(pwa2_4, 0)
		wa2_4 = (t.exp(pwa2_4 - hwa2_4).view(pool_size0 * pool_size0, -1) * wg2).view(pool_size0, -1)
		
		sig_w2_1 = t.sum(wa2_1, 0)
		w2_1 = wa2_1 / sig_w2_1
		sig_w2_2 = t.sum(wa2_2, 0)
		w2_2 = wa2_2 / sig_w2_2
		sig_w2_3 = t.sum(wa2_3, 0)
		w2_3 = wa2_3 / sig_w2_3
		sig_w2_4 = t.sum(wa2_4, 0)
		w2_4 = wa2_4 / sig_w2_4
		
		# concat the blocks
		relation2 = t.cat((t.cat((w2_1.t().mm(Wv2_1), w2_2.t().mm(Wv2_2)), 1), t.cat((w2_3.t().mm(Wv2_3), w2_4.t().mm(Wv2_4)), 1)), 1)
		fc9_input = fc8 + relation2

		fc_cls = self.classifier(fc9_input) 
		
		roi_cls_locs = self.cls_loc(fc_cls)
		roi_scores = self.score(fc_cls)
		return roi_cls_locs, roi_scores
