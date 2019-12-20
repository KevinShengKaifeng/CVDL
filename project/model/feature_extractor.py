import torch as t
from torch import nn
from torchvision.models import vgg16

# to extract features from vgg16 (torchvison pretrained)
def make_extractor_from_torchvision():
	model = vgg16(pretrained=True)
	features = list(model.features)[:30]
# fix the first few layers
	for layer in features[:10]:
		for p in layer.parameters():
			p.requires_grad = False
	return nn.Sequential(*features)
