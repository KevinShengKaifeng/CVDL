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

		# raw plus one linear layer
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
		)
		# location of each class
		self.cls_loc = nn.Linear(4096, n_class * 4)
		# socre of each class
		self.score = nn.Linear(4096, n_class)
		
		self.cls_loc.weight.data.normal_(0, 0.001)
		self.cls_loc.bias.data.zero_()
		self.score.weight.data.normal_(0, 0.01)
		self.score.bias.data.zero_()

		self.n_class = n_class
		self.roi_size = roi_size
		self.spatial_scale = spatial_scale
		# roi pooling layer
		self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

	def forward(self, x, rois, roi_indices):
		# rois from rpn
		roi_indices = at.totensor(roi_indices).float()
		rois = at.totensor(rois).float()
		indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
		# NOTE: important: yx->xy
		xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
		indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

		pool = self.roi(x, indices_and_rois)
		pool = pool.view(pool.size(0), -1)
		# pooling to the same size
		# num_rois * (512 * 7 * 7)
		fc7 = self.classifier(pool)
		roi_cls_locs = self.cls_loc(fc7)
		roi_scores = self.score(fc7)
		return roi_cls_locs, roi_scores

