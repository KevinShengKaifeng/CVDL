import os
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from utils import box_tool
import numpy as np
from utils.config import opt
import xml.etree.ElementTree
from PIL import Image

def resize_image(img, min_size=opt.min_size, max_size=opt.max_size):
	C, H, W = img.shape
	scale = min(min_size / min(H, W), max_size / max(H, W))
	img = img / 255.
	img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
	normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
								std=[0.229, 0.224, 0.225])
	return normalize(t.from_numpy(img)).numpy()


class VOCBboxDataset:
	def __init__(self, split='trainval',
				 use_difficult=False, return_difficult=False,
				 ):
		self.data_dir = 'VOCdevkit/VOC2007/'
		id_list_file = os.path.join(
			self.data_dir, 'ImageSets/Main/{0}.txt'.format(split))
		self.ids = [id_.strip() for id_ in open(id_list_file)]
		self.use_difficult = use_difficult
		self.return_difficult = return_difficult

	def __len__(self):
		return len(self.ids)

	def get_item(self, idx):
		id_ = self.ids[idx]
		anno = xml.etree.ElementTree.parse(
			os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
		bbox = list()
		label = list()
		difficult = list()
		for obj in anno.findall('object'):
			if not self.use_difficult and int(obj.find('difficult').text) == 1:
				continue

			difficult.append(int(obj.find('difficult').text))
			bndbox_anno = obj.find('bndbox')
			# subtract 1 to make pixel indexes 0-based
			bbox.append([
				int(bndbox_anno.find(tag).text) - 1
				for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
			name = obj.find('name').text.lower().strip()
			label.append(opt.VOC_BBOX_LABEL_NAMES.index(name))
		bbox = np.stack(bbox).astype(np.float32)
		label = np.stack(label).astype(np.int32)
		difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

		img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
		file = Image.open(img_file)
		img = np.asarray(file.convert('RGB'), dtype=np.float32)
		if img.ndim == 2:
			img = img[np.newaxis]
		else:
			img = img.transpose((2, 0, 1))

		return img, bbox, label, difficult
	__getitem__ = get_item

class Dataset:
	def __init__(self):
		self.vocdataset = VOCBboxDataset()

	def __getitem__(self, idx):
		ori_img, bbox, label, difficult = self.vocdataset.get_item(idx)

		_, H, W = ori_img.shape
		img = resize_image(ori_img)
		_, o_H, o_W = img.shape
		scale = o_H / H
		bbox = box_tool.resize_bbox(bbox, (H, W), (o_H, o_W))

		img, params = box_tool.random_flip(
			img, x_random=True, return_param=True)
		bbox = box_tool.flip_bbox(
			bbox, (o_H, o_W), x_flip=params['x_flip'])
		
		return img.copy(), bbox.copy(), label.copy(), scale

	def __len__(self):
		return len(self.vocdataset)


class TestDataset:
	def __init__(self, use_difficult=True):
		self.vocdataset = VOCBboxDataset(split='test', use_difficult=use_difficult)

	def __getitem__(self, idx):
		ori_img, bbox, label, difficult = self.vocdataset.get_item(idx)
		img = resize_image(ori_img)
		return img, ori_img.shape[1:], bbox, label, difficult

	def __len__(self):
		return len(self.vocdataset)
