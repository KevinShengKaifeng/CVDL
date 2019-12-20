class Config:
	# data
	min_size = 600 # image resize
	max_size = 1000 # image resize

	# train
	weight_decay = 0.0005
	lr_decay = 0.1
	lr = 1e-3
	epoch = 18

	# visualization
	env = 'fasterrcnn'  # visdom env
	port = 8097
	plot_every = 50  # vis every N iter
	load_path = None #'checkpoints/fasterrcnn_raw_06121344_170.686274260029252'

	VOC_BBOX_LABEL_NAMES = (
		'aeroplane',
		'bicycle',
		'bird',
		'boat',
		'bottle',
		'bus',
		'car',
		'cat',
		'chair',
		'cow',
		'diningtable',
		'dog',
		'horse',
		'motorbike',
		'person',
		'pottedplant',
		'sheep',
		'sofa',
		'train',
		'tvmonitor')


opt = Config()
