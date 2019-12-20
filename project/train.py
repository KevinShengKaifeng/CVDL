import os
import fire
import time
from tqdm import tqdm
from collections import namedtuple
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils import data
from utils.config import opt
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from model import FasterRCNN
from dataset import Dataset, TestDataset
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from utils.vis_tool import Visualizer
from torchnet.meter import AverageValueMeter
from trainer import FasterRCNNTrainer
import matplotlib
matplotlib.use('agg')

def eval(dataloader, faster_rcnn, test_num=1500):
	pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults  = list(), list(), list(), list(), list(), list()
	for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
		sizes = [sizes[0][0], sizes[1][0]]
		pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs[0], sizes)
		gt_bboxes += list(gt_bboxes_.numpy())
		gt_labels += list(gt_labels_.numpy())
		gt_difficults += list(gt_difficults_.numpy())
		pred_bboxes += pred_bboxes_
		pred_labels += pred_labels_
		pred_scores += pred_scores_
		if ii == test_num: break

	result = eval_detection_voc(
		pred_bboxes, pred_labels, pred_scores,
		gt_bboxes, gt_labels, gt_difficults,
		)
	return result

def train(**kwargs):
	dataset = Dataset()
	print('load data')
	dataloader = t.utils.data.DataLoader(dataset, batch_size=1, shuffle = True)
	testset = TestDataset()
	test_dataloader = t.utils.data.DataLoader(testset, batch_size=1)
	faster_rcnn = FasterRCNN()
	print('model construct completed')
	trainer = FasterRCNNTrainer(faster_rcnn).cuda()
	if opt.load_path:
		state_dict = t.load(opt.load_path)
		trainer.faster_rcnn.load_state_dict(state_dict)
		print('load pretrained model from %s' % opt.load_path)

	best_map = 0
	for epoch in range(opt.epoch):
		trainer.reset_meters()
		
		for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
			scale = at.scalar(scale)
			img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
			img, bbox, label = Variable(img), Variable(bbox), Variable(label)
			trainer.train_step(img, bbox, label, scale)

			if (ii + 1) % opt.plot_every == 1:
				# plot loss
				trainer.vis.plot_many(trainer.get_meter_data())

			if (ii + 1) % opt.plot_every == 1:
				# plot groud truth bboxes
				ori_img_ = (at.tonumpy(img[0]) * 0.225 + 0.45).clip(min=0, max=1) * 255
				gt_img = visdom_bbox(ori_img_,
									 at.tonumpy(bbox_[0]),
									 at.tonumpy(label_[0]))
				trainer.vis.img('gt_img', gt_img)
				# plot predicti bboxes
				_bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
				pred_img = visdom_bbox(ori_img_,
									   at.tonumpy(_bboxes[0]),
									   at.tonumpy(_labels[0]).reshape(-1),
									   at.tonumpy(_scores[0]))
				trainer.vis.img('pred_img', pred_img)
				
		# learning rate decay	
		if epoch == 9:
			for param_group in trainer.faster_rcnn.optimizer.param_groups:
				param_group['lr'] *= opt.lr_decay
		# evaluate and save checkpoints
		if epoch >= 9:
			eval_result = eval(test_dataloader, faster_rcnn, test_num=5000)
			print(eval_result['map'])
			if eval_result['map'] > best_map:
				best_map = eval_result['map']
			timestr = time.strftime('%m%d%H%M')
			save_path = 'checkpoints/fasterrcnn_%s' % timestr
			save_path = save_path + '_' + str(epoch) + '_' + str(eval_result['map'])
			t.save(trainer.faster_rcnn.state_dict(), save_path)
		if epoch == opt.epoch - 1: 
			print(best_map)
			break

if __name__ == '__main__':
	fire.Fire()
