import random
import numpy as np
import re
from PIL import Image, ImageFilter
import time
import threading
import matplotlib.pyplot as plt


DATA_AUGMENT = True


def image_clip(image):
    strip = 80
    if image.width >= image.height:
        edge = image.height
        clips = []
        while edge < image.width:
            clips.append(image.crop((edge-image.height, 0, edge, image.height)))
            edge += strip
        return clips
    else:
        edge = image.width
        clips = []
        while edge < image.height:
            clips.append(image.crop((0, edge-image.width, image.width, edge)))
            edge += strip
        return clips


def load_data(description, augment=False):
    pair = re.split(' ', description)
    sy = np.zeros(80)
    sy[int(pair[1])] = 1
    with Image.open("data/"+pair[0]) as im:
        if not augment:
            return [[np.array(im.resize((256, 256), Image.ANTIALIAS)), sy]]
        else:
            im_array = []
            im_array.append(im)
            im_array += image_clip(im)
            im_array.append(im.transpose(Image.FLIP_LEFT_RIGHT))
            im_array += image_clip(im.transpose(Image.FLIP_LEFT_RIGHT))
            im_array.append(im.filter(ImageFilter.BLUR))
            data_array = []
            for ima in im_array:
                data_array.append([np.array(ima.resize((256, 256), Image.ANTIALIAS)), sy])
            return data_array


def initialize():
    global test_data, t
    print("initializing data sets...")
    for i in train_info[-1000:]:
        test_data += load_data(i)
    for ti in t:
        ti.start()
    refresh_data()


def buffer_data(size, buff_n):
    global counter, train_data_buffer
    train_data_buffer[buff_n] = []
    lock.acquire()
    s = counter % 55000
    counter += size
    lock.release()
    for i in train_info[s:min(s + size, len(train_info)-1000)]:
        train_data_buffer[buff_n] += load_data(i, DATA_AUGMENT)


def refresh_data():
    global train_data, t
    for ti in t:
        ti.join()
    print("refreshing training data, data set N.%d" % (counter // (buffer_size*len(t))))
    train_data = []
    for i in train_data_buffer:
        train_data += i
    print("time: %.2fmin" % ((time.time() - stime) / 60))
    for i in range(buffer_num):
        t[i] = threading.Thread(target=buffer_data, args=(buffer_size, i))
        t[i].start()


def next_batch(size, iftest=False):
    batch = []
    if size > 1000:
        raise ValueError("Batch too large!")
    if iftest:
        batch = test_data[:size]
    else:
        for i in range(size):
            sample = train_data[random.randint(0, len(train_data)-1)]
            batch.append(sample)
#    plt.imshow(batch[0][0])
#    plt.show()
    batch = np.array(batch).T
    return np.array(batch[0].tolist()), np.array(batch[1].tolist())


train_data = []
test_data = []
counter = 0
stime = time.time()
lock = threading.Lock()
t = []
train_data_buffer = []
if DATA_AUGMENT:
    buffer_size = 1000
else:
    buffer_size = 5000
buffer_num = 2
for i in range(buffer_num):
    train_data_buffer.append([])
    t.append(threading.Thread(target=buffer_data, args=(buffer_size, i)))
with open("data/train.info", 'r') as train_file:
    train_info = train_file.readlines()
initialize()
