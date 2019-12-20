import os
import torch as t
from model import FasterRCNN
from trainer import FasterRCNNTrainer
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')

import numpy as np

# input 1.jpg
# output ans.jpg
file = Image.open('1.jpg')
img = np.asarray(file.convert('RGB'), dtype=np.float32)
if img.ndim == 2:
	img = img[np.newaxis]
else:
	img = img.transpose((2, 0, 1))
img = t.from_numpy(img)[None]
faster_rcnn = FasterRCNN()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
# use parameters for the raw Faster RCNN model
state_dict = t.load('checkpoints/fasterrcnn_raw_06121344_170.686274260029252')
trainer.faster_rcnn.load_state_dict(state_dict)

img = img[0]
_bboxes, _labels, _scores = trainer.faster_rcnn.predict([at.tonumpy(img)], visualize=True)
vis_bbox(at.tonumpy(img),
		at.tonumpy(_bboxes[0]),
		at.tonumpy(_labels[0]).reshape(-1),
		at.tonumpy(_scores[0]).reshape(-1))
