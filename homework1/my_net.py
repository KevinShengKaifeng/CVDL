import tensorflow as tf
from data_augmentation import next_batch, refresh_data
import time
import numpy as np
import matplotlib.pyplot as plt
from test_output import load_test_info


def weight_variable(shape):
    global weight_array, loading_number
    if load_pre_train:
        weight = tf.Variable(weight_array[loading_number])
        weight_array[loading_number] = weight
    else:
        initial = tf.truncated_normal(shape, stddev=np.sqrt(2/shape[0]/shape[1]/shape[2]))
        weight = tf.Variable(initial)
        weight_array.append(weight)
    return weight


def fc_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2/shape[0]))
    return tf.Variable(initial)


def bias_variable(shape):
    global bias_array, loading_number
    if load_pre_train:
        bias = tf.Variable(bias_array[loading_number])
        bias_array[loading_number] = bias
        loading_number += 1
    else:
        initial = tf.constant(1.0, shape=shape)
        bias = tf.Variable(initial)
        bias_array.append(bias)
    return bias


def conv2d(x, i, o, size=3, stride=1):
    global weight_array
    W = weight_variable([size, size, i, o])
    b = bias_variable([o])
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')+b


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def make_layers_plain(x, n):
    filters = int(x.shape[-1])
    layers = [x]
    for i in range(n):
        layers.append(tf.nn.relu(conv2d(batch_norm(layers[-1]), filters, filters)))
    return layers[-1]


def make_layers(x, n, enlarge=True):
    assert n % 2 == 0
    layers = [x]
    global batch_size
    if enlarge:
        filters = 2*int(x.shape[-1])
        x_p = max_pool(x)
        W = tf.constant(0.0, shape=[1, 1, filters//2, filters//2])
        x_zeros = tf.nn.conv2d(x_p, W, strides=[1,1,1,1], padding="SAME")
        x_l = tf.concat([x_p, x_zeros], 3)
        layers.append(conv2d(batch_norm(tf.nn.relu(layers[-1])), filters//2, filters, stride=2))
        layers.append(conv2d(batch_norm(tf.nn.relu(layers[-1])), filters, filters)+x_l)
        for i in range(n//2-1):
            layers.append(conv2d(batch_norm(tf.nn.relu(layers[-1])), filters, filters))
            layers.append(conv2d(batch_norm(tf.nn.relu(layers[-1])), filters, filters) + layers[-2])
    else:
        filters = int(x.shape[-1])
        for i in range(n//2):
            layers.append(conv2d(batch_norm(tf.nn.relu(layers[-1])), filters, filters))
            layers.append(conv2d(batch_norm(tf.nn.relu(layers[-1])), filters, filters) + layers[-2])
    return layers[-1]


def fc(x, i, o):
    global weight_array
    W = fc_weight_variable([i, o])
    weight_array.append(W)
    b = bias_variable([o])
    return tf.matmul(x, W)+b


def batch_norm(x):
    shape = x.shape[1:]
    x_mean, x_var = tf.nn.moments(x, [0])
    offset = tf.Variable(tf.constant(0.0, shape=shape))
    scale = tf.Variable(tf.constant(1.0, shape=shape))
    return tf.nn.batch_normalization(x, x_mean, x_var, offset, scale, 0.001)


def resnet_34_plain(x):
    h_conv1_input = max_pool(tf.nn.relu(conv2d(batch_norm(x), 3, 64, size=7, stride=2)))
    h_conv1_output = make_layers_plain(h_conv1_input, 6)
    h_conv2_input = tf.nn.relu(conv2d(batch_norm(h_conv1_output), 64, 128, stride=2))
    h_conv2_output = make_layers_plain(h_conv2_input, 7)
    h_conv3_input = tf.nn.relu(conv2d(batch_norm(h_conv2_output), 128, 256, stride=2))
    h_conv3_output = make_layers_plain(h_conv3_input, 11)
    h_conv4_input = tf.nn.relu(conv2d(batch_norm(h_conv3_output), 256, 512, stride=2))
    h_conv4_output = make_layers_plain(h_conv4_input, 5)
    feature_map = tf.nn.relu(conv2d(batch_norm(h_conv4_output), 512, 80, size=1))
    global_pool = tf.reduce_mean(feature_map, [1, 2])
    y = tf.nn.softmax(tf.reshape(global_pool, [-1, 80]))
    return y


def resnet_34(x):
    h_conv1_input = max_pool(tf.nn.relu(conv2d(batch_norm(x), 3, 64, size=7, stride=2)))
    h_conv1_output = make_layers(h_conv1_input, 6, enlarge=False)
    h_conv2_output = make_layers(h_conv1_output, 8)
    h_conv3_output = make_layers(h_conv2_output, 12)
    h_conv4_output = make_layers(h_conv3_output, 6)
    feature_map = conv2d(batch_norm(tf.nn.relu(h_conv4_output)), 512, 80, size=1)
    global_pool = tf.reduce_mean(feature_map, [1, 2])
    y = tf.nn.softmax(tf.reshape(global_pool, [-1, 80]))
    return y


stime = time.time()
x = tf.placeholder(tf.float32, [None, 256, 256, 3])
y_ = tf.placeholder("float", [None, 80])
sess = tf.InteractiveSession()
load_pre_train = False
save_pre_train = True
if load_pre_train:
    weight_array = np.load("weight.npy")
    bias_array = np.load("bias.npy")
else:
    weight_array = []
    bias_array = []
loading_number = 0
y_conv = resnet_34(x)
alpha = 0.01
loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))\
       + alpha*sum([tf.reduce_sum(w*w) for w in weight_array])
train3 = tf.train.AdamOptimizer(1e-3).minimize(loss)
train4 = tf.train.AdamOptimizer(1e-4).minimize(loss)
train5 = tf.train.AdamOptimizer(1e-5).minimize(loss)
train6 = tf.train.AdamOptimizer(1e-6).minimize(loss)
correct_prediction1 = tf.nn.in_top_k(y_conv, tf.argmax(y_, 1), 1)
correct_prediction3 = tf.nn.in_top_k(y_conv, tf.argmax(y_, 1), 3)
accuracy = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, "float"))
sess.run(tf.global_variables_initializer())
step = 9900
batch_size = 100
data_size = 55000
train_acc = []
test_acc = []
for i in range(step):
    batch = next_batch(batch_size)
    train_accuracy = 0
    if not load_pre_train:
        if i < data_size//batch_size*3:
            train3.run(feed_dict={x: batch[0], y_: batch[1]})
        elif data_size//batch_size*3 <= i < data_size//batch_size*6:
            train4.run(feed_dict={x: batch[0], y_: batch[1]})
        elif data_size//batch_size*6 <= i < data_size//batch_size*9:
            train5.run(feed_dict={x: batch[0], y_: batch[1]})
        else:
            train5.run(feed_dict={x: batch[0], y_: batch[1]})
    else:
        train5.run(feed_dict={x: batch[0], y_: batch[1]})
    if (i+1) % 10 == 0:
        loss_value, train_accuracy = sess.run((loss, accuracy), feed_dict={x: batch[0], y_: batch[1]})
        print("step %d, training top1 accuracy %g, loss %f" % (i+1, train_accuracy, loss_value))
        train_acc.append([train_accuracy, i*batch_size/data_size])
    if (i+1) % 50 == 0:
        batch = next_batch(1000, iftest=True)
        test_accuracy1 = (accuracy.eval(feed_dict={x: batch[0][:250], y_: batch[1][:250]}) +
                accuracy.eval(feed_dict={x: batch[0][250:500], y_: batch[1][250:500]}) +
                accuracy.eval(feed_dict={x: batch[0][500:-250], y_: batch[1][500:-250]}) +
                accuracy.eval(feed_dict={x: batch[0][-250:], y_: batch[1][-250:]})) / 4
        test_accuracy3 = (accuracy3.eval(feed_dict={x: batch[0][:250], y_: batch[1][:250]}) +
                 accuracy3.eval(feed_dict={x: batch[0][250:500], y_: batch[1][250:500]}) +
                 accuracy3.eval(feed_dict={x: batch[0][500:-250], y_: batch[1][500:-250]}) +
                 accuracy3.eval(feed_dict={x: batch[0][-250:], y_: batch[1][-250:]})) / 4
        print("test top1 accuracy %.3f, top3 accuracy %.3f" % (test_accuracy1, test_accuracy3))
        print("training time: %.2fmin" % ((time.time() - stime) / 60))
        test_acc.append([test_accuracy1, test_accuracy3, i*batch_size/data_size])
        fig = plt.figure("accuracy plot")
        plt.plot(np.array(train_acc).T[1], np.array(train_acc).T[0])
        plt.plot(np.array(test_acc).T[2], np.array(test_acc).T[0])
        plt.plot(np.array(test_acc).T[2], np.array(test_acc).T[1])
        fig.savefig(str(int(stime))+"resnet_34")
    if (i + 1) % 50 == 0:
        refresh_data()
        if save_pre_train:
            assert len(weight_array) == len(bias_array)
            nweight_array = []
            nbias_array = []
            for i in range(len(weight_array)):
                nweight_array.append(weight_array[i].eval())
                nbias_array.append(bias_array[i].eval())
            np.save("weight.npy", nweight_array)
            np.save("bias.npy", nbias_array)

test_pics, output_str = load_test_info()
for i in range(20):
    predic = y_conv.eval(feed_dict={x: test_pics[250*i:250*(i+1)]})
    for j in range(len(predic)):
        c = np.argsort(predic[j])[-3:][::-1]
        output_str[i * 250 + j] = output_str[i*250+j][:-1]
        output_str[i * 250 + j] += " "+str(c[0])+" "+str(c[1])+" "+str(c[2])+"\n"
with open("submit.info", 'w') as sub:
    for line in output_str:
        sub.write(line)
