from collections import namedtuple
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from torch.autograd import Variable
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import AverageValueMeter

LossTuple = namedtuple('LossTuple',
					   ['rpn_loc_loss',
						'rpn_cls_loss',
						'roi_loc_loss',
						'roi_cls_loss',
						'total_loss'
						])


class FasterRCNNTrainer(nn.Module):
	"""wrapper for conveniently training. return losses

	The losses include:

	* :obj:`rpn_loc_loss`: The localization loss for \
		Region Proposal Network (RPN).
	* :obj:`rpn_cls_loss`: The classification loss for RPN.
	* :obj:`roi_loc_loss`: The localization loss for the head module.
	* :obj:`roi_cls_loss`: The classification loss for the head module.
	* :obj:`total_loss`: The sum of 4 loss above.

	Args:
		faster_rcnn (model.FasterRCNN):
			A Faster R-CNN model that is going to be trained.
	"""

	def __init__(self, faster_rcnn):
		super(FasterRCNNTrainer, self).__init__()

		self.faster_rcnn = faster_rcnn
		self.rpn_sigma = 3.0
		self.roi_sigma = 1.0

		# target creator create gt_bbox gt_label etc as training targets. 
		self.anchor_target_creator = AnchorTargetCreator()
		self.proposal_target_creator = ProposalTargetCreator()

		self.loc_normalize_mean = (0., 0., 0., 0.)
		self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

		self.optimizer = self.faster_rcnn.optimizer
		# visdom wrapper
		self.vis = Visualizer(env=opt.env)

		# indicators for training status
		self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

	def forward(self, imgs, bboxes, labels, scale):
		"""Forward Faster R-CNN and calculate losses.

		Here are notations used.

		* :math:`N` is the batch size.
		* :math:`R` is the number of bounding boxes per image.

		Currently, only :math:`N=1` is supported.

		Args:
			imgs (~torch.autograd.Variable): A variable with a batch of images.
			bboxes (~torch.autograd.Variable): A batch of bounding boxes.
				Its shape is :math:`(N, R, 4)`.
			labels (~torch.autograd..Variable): A batch of labels.
				Its shape is :math:`(N, R)`. The background is excluded from
				the definition, which means that the range of the value
				is :math:`[0, L - 1]`. :math:`L` is the number of foreground
				classes.
			scale (float): Amount of scaling applied to
				the raw image during preprocessing.

		Returns:
			namedtuple of 5 losses
		"""
		n = bboxes.shape[0]

		_, _, H, W = imgs.shape
		img_size = (H, W)

		features = self.faster_rcnn.extractor(imgs)

		rpn_locs, rpn_scores, rois, roi_indices, anchor = \
			self.faster_rcnn.rpn(features, img_size, scale)

		bbox = bboxes[0]
		label = labels[0]
		rpn_score = rpn_scores[0]
		rpn_loc = rpn_locs[0]
		roi = rois
		
		sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
			roi,
			at.tonumpy(bbox),
			at.tonumpy(label),
			self.loc_normalize_mean,
			self.loc_normalize_std)
		sample_roi_index = t.zeros(len(sample_roi))
		roi_cls_loc, roi_score = self.faster_rcnn.head(
			features,
			sample_roi,
			sample_roi_index)

		# ------------------ RPN losses -------------------#
		gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
			at.tonumpy(bbox),
			anchor,
			img_size)
		gt_rpn_label = at.tovariable(gt_rpn_label).long()
		gt_rpn_loc = at.tovariable(gt_rpn_loc)
		rpn_loc_loss = _fast_rcnn_loc_loss(
			rpn_loc,
			gt_rpn_loc,
			gt_rpn_label.data,
			self.rpn_sigma)

		rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
		_gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
		_rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]

		# ------------------ ROI losses (fast rcnn loss) -------------------#
		n_sample = roi_cls_loc.shape[0]
		roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
		roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
							  at.totensor(gt_roi_label).long()]
		gt_roi_label = at.tovariable(gt_roi_label).long()
		gt_roi_loc = at.tovariable(gt_roi_loc)

		roi_loc_loss = _fast_rcnn_loc_loss(
			roi_loc.contiguous(),
			gt_roi_loc,
			gt_roi_label.data,
			self.roi_sigma)

		roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

		losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
		losses = losses + [sum(losses)]

		return LossTuple(*losses)

	def train_step(self, imgs, bboxes, labels, scale):
		self.optimizer.zero_grad()
		losses = self.forward(imgs, bboxes, labels, scale)
		losses.total_loss.backward()
		self.optimizer.step()
		self.update_meters(losses)
		return losses

	def update_meters(self, losses):
		loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
		for key, meter in self.meters.items():
			meter.add(loss_d[key])

	def reset_meters(self):
		for key, meter in self.meters.items():
			meter.reset()

	def get_meter_data(self):
		return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
	sigma2 = sigma ** 2
	diff = in_weight * (x - t)
	abs_diff = diff.abs()
	flag = (abs_diff.data < (1. / sigma2)).float()
	flag = Variable(flag)
	y = (flag * (sigma2 / 2.) * (diff ** 2) +
		 (1 - flag) * (abs_diff - 0.5 / sigma2))
	return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
	in_weight = t.zeros(gt_loc.shape).cuda()
	in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
	loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
	loc_loss /= (gt_label >= 0).sum()
	return loc_loss
