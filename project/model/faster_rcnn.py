import torch as t
import cupy as cp
import numpy as np
from torch import nn
from utils.config import opt
from dataset import resize_image
from utils import array_tool as at
from torch.nn import functional as F
from model.fast_rcnn import FastRCNN
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression
from model.region_proposal_network import RegionProposalNetwork
from model.feature_extractor import make_extractor_from_torchvision

class FasterRCNN(nn.Module):
	def __init__(self,
				 n_fg_class=20,
				 ):
		super(FasterRCNN, self).__init__()
		# faster rcnn is composed of vgg16extractor, FastRCNN and RPN
		self.extractor = make_extractor_from_torchvision()
		self.head = FastRCNN()
		self.rpn = RegionProposalNetwork(
			512, 512,
			ratios=[0.5, 1, 2],
			anchor_scales=[8, 16, 32],
			feat_stride=16,
		)
		# initialize optimizer
		params = []
		for key, value in dict(self.named_parameters()).items():
			if value.requires_grad:
				if 'bias' in key:
					params += [{'params': [value], 'lr': opt.lr * 2, 'weight_decay': 0}]
				else:
					params += [{'params': [value], 'lr': opt.lr, 'weight_decay': opt.weight_decay}]
		self.optimizer = t.optim.SGD(params, momentum=0.9)
		self.prob = 0.05

	# forward the Faster RCNN
	def forward(self, x, scale=1.):
		img_size = x.shape[2:]
		feature = self.extractor(x)
		rpn_locs, rpn_scores, rois, roi_indices, anchor = \
			self.rpn(feature, img_size, scale)
		roi_cls_locs, roi_scores = self.head(feature, rois, roi_indices)
		return roi_cls_locs, roi_scores, rois, roi_indices
	
	def _suppress(self, raw_cls_bbox, raw_prob):
		bbox, label, score = list(), list(), list()
		for l in range(1, 21):
			cls_bbox_l = raw_cls_bbox.reshape((-1, 21, 4))[:, l, :]
			prob_l = raw_prob[:, l]
			mask = prob_l > self.prob
			cls_bbox_l = cls_bbox_l[mask]
			prob_l = prob_l[mask]
			keep = non_maximum_suppression(
				cp.array(cls_bbox_l), 0.3, prob_l)
			keep = cp.asnumpy(keep)
			bbox.append(cls_bbox_l[keep])
			label.append((l - 1) * np.ones((len(keep),)))
			score.append(prob_l[keep])
		bbox = np.concatenate(bbox, axis=0).astype(np.float32)
		label = np.concatenate(label, axis=0).astype(np.int32)
		score = np.concatenate(score, axis=0).astype(np.float32)
		return bbox, label, score	
		
	def predict(self, img, size=None,visualize=False):
		loc_normalize_mean = (0., 0., 0., 0.)
		loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
		self.eval() # convert to eval model
		if visualize:
			size = img[0].shape[1:]
			img = resize_image(at.tonumpy(img[0]))
			self.prob = 0.7
		bboxes, labels, scores = list(), list(), list()
		img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
		scale = img.shape[3] / size[1]
		roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
		roi_score = roi_scores.data
		roi_cls_loc = roi_cls_loc.data
		roi = at.totensor(rois) / scale

		# Convert predictions to bounding boxes in image coordinates.
		# Bounding boxes are scaled to the scale of the input images.
		mean = t.Tensor(loc_normalize_mean).cuda(). \
			repeat(21)[None]
		std = t.Tensor(loc_normalize_std).cuda(). \
			repeat(21)[None]
		roi_cls_loc = (roi_cls_loc * std + mean)
		roi_cls_loc = roi_cls_loc.view(-1, 21, 4)
		roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
		cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
							at.tonumpy(roi_cls_loc).reshape((-1, 4)))
		cls_bbox = at.totensor(cls_bbox)
		cls_bbox = cls_bbox.view(-1, 21 * 4)
		# clip bounding box
		cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
		cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

		prob = at.tonumpy(F.softmax(at.tovariable(roi_score), dim=1))

		raw_cls_bbox = at.tonumpy(cls_bbox)
		raw_prob = at.tonumpy(prob)

		bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
		bboxes.append(bbox)
		labels.append(label)
		scores.append(score)

		self.train() # return to train model
		self.prob = 0.05
		return bboxes, labels, scores
