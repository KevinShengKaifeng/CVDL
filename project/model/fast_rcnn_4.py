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
		# four blocks
		self.make_Wk1_1 = nn.Linear(4096, 4096)
		self.make_Wq1_1 = nn.Linear(4096, 4096)
		self.make_Wv1_1 = nn.Linear(4096, 1024)
		self.make_Wk1_2 = nn.Linear(4096, 4096)
		self.make_Wq1_2 = nn.Linear(4096, 4096)
		self.make_Wv1_2 = nn.Linear(4096, 1024)
		self.make_Wk1_3 = nn.Linear(4096, 4096)
		self.make_Wq1_3 = nn.Linear(4096, 4096)
		self.make_Wv1_3 = nn.Linear(4096, 1024)
		self.make_Wk1_4 = nn.Linear(4096, 4096)
		self.make_Wq1_4 = nn.Linear(4096, 4096)
		self.make_Wv1_4 = nn.Linear(4096, 1024)
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

		# pooling to the same size
		# num_rois * (512 * 7 * 7)
		pool = self.roi(x, indices_and_rois)
		pool_size0 = pool.size(0)
		pool = pool.view(pool_size0, -1)
		
		ps_one = t.zeros(pool_size0, pool_size0, 4).cuda()
	
		# one layer
		# four blocks without location info
		# compute wa and w
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
		
		pwa1_1 = Wk1_1.mm(Wq1_1.t()) / 64
		hwa1_1, _ = t.max(pwa1_1, 0)
		wa1_1 = t.exp(pwa1_1 - hwa1_1)
		sig_w1_1 = t.sum(wa1_1, 0)
		w1_1 = wa1_1 / sig_w1_1
		
		pwa1_2 = Wk1_2.mm(Wq1_2.t()) / 64
		hwa1_2, _ = t.max(pwa1_2, 0)
		wa1_2 = t.exp(pwa1_2 - hwa1_2)
		sig_w1_2 = t.sum(wa1_2, 0)
		w1_2 = wa1_2 / sig_w1_2
		
		pwa1_3 = Wk1_3.mm(Wq1_3.t()) / 64 
		hwa1_3, _ = t.max(pwa1_3, 0)
		wa1_3 = t.exp(pwa1_3 - hwa1_3)
		sig_w1_3 = t.sum(wa1_3, 0)
		w1_3 = wa1_3 / sig_w1_3
		
		pwa1_4 = Wk1_4.mm(Wq1_4.t()) / 64 
		hwa1_4, _ = t.max(pwa1_4, 0)
		wa1_4 = t.exp(pwa1_4 - hwa1_4)
		sig_w1_4 = t.sum(wa1_4, 0)
		w1_4 = wa1_4 / sig_w1_4
		
		# concat the blocks
		relation = t.cat((t.cat((w1_1.t().mm(Wv1_1), w1_2.t().mm(Wv1_2)), 1), t.cat((w1_3.t().mm(Wv1_3), w1_4.t().mm(Wv1_4)), 1)), 1)
		fc8_input = fc7	+ relation
		
		fc_cls = self.classifier(fc8_input) 
		
		roi_cls_locs = self.cls_loc(fc_cls)
		roi_scores = self.score(fc_cls)
		return roi_cls_locs, roi_scores
