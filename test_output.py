from PIL import Image, ImageFilter
import numpy as np


def load_test_info():
    with open("data/test.info", 'r') as test_file:
        test_info = test_file.readlines()
    test_pics = []
    for pic in test_info:
        with Image.open("data/"+pic[:-1]) as im:
            test_pics.append(np.array(im.resize((256, 256), Image.ANTIALIAS)))
    return test_pics, test_info
