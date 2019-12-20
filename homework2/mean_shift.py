from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import threading


def shift(point, pic_array, kernel_type):
    global radius
    if kernel_type:
        w = np.exp((-np.sum((pic_array-point)**2, 2)/radius**2))
    else:
        w = (np.sum((pic_array-point)**2, 2) < radius**2)
        w = w.astype(np.int32)
    w = w.reshape((w.shape[0], w.shape[1], 1))/np.sum(w)
    s = np.sum(np.sum(w*pic_array, 0), 0)
    return s


def binding_coordinate(pic):
    pic_data = pic.load()
    pic_array = []
    for x in range(pic.size[0]):
        line = []
        for y in range(pic.size[1]):
            line.append(
                np.array(list(pic_data[x, y]) + [x / pic.size[0] * 256 * alpha] + [y / pic.size[1] * 256 * alpha]))
        pic_array.append(line)
    return np.array(pic_array)


def segment(pic_array):
    peak_set = [np.zeros(5)]
    peak_distribution = np.zeros(pic.size) - 1
    stime = time.time()
    thread_array = []
    lock = threading.Lock()

    for x in range(0, pic.size[0]):
        def compute_column(x):
            for y in range(0, pic.size[1]):
                p = pic_array[x, y]
                for step in range(max_step):
                    next_p = shift(p, pic_array, kernel_type)
                    #w = np.array([np.sum((peak - next_p) ** 2) < 1 for peak in peak_set])
                    #if w.any():
                    #    c = w.tolist().index(True)
                    #    break
                    if np.sum((p-next_p)**2) < 0.01:
                        break
                    p = next_p
                lock.acquire()
                peak_set.append(p)
                c = len(peak_set) - 1
                global counter
                counter += 1
                if counter % pic.size[1] == 0:
                    print(counter // pic.size[1])
                    print(time.time() - stime)
                lock.release()
                peak_distribution[x, y] = c

        thread_array.append(threading.Thread(target=compute_column, args=(x,)))
        thread_array[-1].start()
    for t in thread_array:
        t.join()
    np.save("set", peak_set)
    np.save("dis", peak_distribution)
    peak_dis = peak_distribution.T
    new_dis = np.zeros((peak_dis.shape[0], peak_dis.shape[1], 3))
    for x in range(peak_dis.shape[0]):
        for y in range(peak_dis.shape[1]):
            new_dis[x, y, 2] = peak_set[int(peak_dis[x, y])][2]
            new_dis[x, y, :2] = peak_set[int(peak_dis[x, y])][:2]
    plt.imshow(256-new_dis)
    plt.savefig("results/2_k%s_r%s_alpha%s.png"%(kernel_type, radius, alpha))


pic = Image.open("pic2.jpg").resize((200, 150), Image.ANTIALIAS)
max_step = 100
for r in (5, 10, 20, 30, 50, 100):
    for a in (0.0, 0.1, 0.3, 0.5, 1.0):
        for k in (1, 0):
            counter = 0
            radius = r
            alpha = a
            kernel_type = k
            segment(binding_coordinate(pic))
