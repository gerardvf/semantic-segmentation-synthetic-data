import os
import glob
import tensorflow as tf

from models import ERFNet


def read_image(path):  # read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # uint8
    return image  # (512, 1024, 3)


num_classes = 19
M_net1 = ERFNet(num_classes)
M_net2 = ERFNet(num_classes)
M_net3 = ERFNet(num_classes)
_ = M_net1(tf.random.normal([1, 512, 1024, 3]), training=False)
_ = M_net2(tf.random.normal([1, 512, 1024, 3]), training=False)
_ = M_net3(tf.random.normal([1, 512, 1024, 3]), training=False)
M_net1.load_weights('./raid_storage/weightsfda1/w12.h5')
M_net2.load_weights('./raid_storage/weightsfda2/w10.h5')
M_net3.load_weights('./raid_storage/weightsfda3/w17.h5')

w1, w2, w3 = 0.5, 0.2, 0.3


data_path = './raid_storage/data'
cityscapes_paths = sorted(glob.glob(f'{data_path}/cityscapes/images/*.jpg'))
with open(f'{data_path}/cityscapes_splits.txt') as f:
    lines = f.readlines()
cityscapes_paths = [cityscapes_paths[int(id)] for id in lines[0].strip().split()[1:]]  # cityscapes training images


# PHASE 1: determine median probability for every class (only using predictions on cityscapes training set)
num_bins = 100
class_histograms = [tf.zeros(num_bins, tf.int32)] * num_classes  # an histogram (list of 30 values representing the number of pixels in a certain bin) for every class

for i, path in enumerate(cityscapes_paths):
    x = tf.cast(read_image(path), tf.float32) / 127.5 - 1.0
    x = tf.expand_dims(x, axis=0)  # (1, 512, 1024, 3)
    
    p_pred1 = M_net1(x, training=False)  # (1, 512, 1024, 19)
    p_pred2 = M_net2(x, training=False)
    p_pred3 = M_net3(x, training=False)
    p_pred1 = tf.nn.softmax(p_pred1, axis=-1)
    p_pred2 = tf.nn.softmax(p_pred2, axis=-1)
    p_pred3 = tf.nn.softmax(p_pred3, axis=-1)
    p_pred = w1*p_pred1 + w2*p_pred2 + w3*p_pred3  # (1, 512, 1024, 19)  the 19 probabilities for every pixel add up to 1
    
    p_max = tf.reduce_max(p_pred, axis=-1)  # (1, 512, 1024)  probability of most probable class for every pixel
    p_argmax = tf.math.argmax(p_pred, axis=-1, output_type=tf.int32)  # (1, 512, 1024)  most probable class for every pixel
    p_max = tf.reshape(p_max, [-1])
    p_argmax = tf.reshape(p_argmax, [-1])
    
    for class_id in range(num_classes):
        class_probs = tf.boolean_mask(p_max, p_argmax == class_id)
        class_histograms[class_id] += tf.histogram_fixed_width(class_probs, [0., 1.], nbins=num_bins)

    if (i + 1) % 100 == 0:
        print(f'[{i+1}/{len(cityscapes_paths)}]')


estimated_median_probs = []
for class_id, histogram in enumerate(class_histograms):
    cumhistogram = tf.math.cumsum(histogram, axis=0)
    median_prob = float(tf.where(cumhistogram >= cumhistogram[-1] // 2)[0]) / num_bins
    estimated_median_probs.append(median_prob)
print(estimated_median_probs)

# include also all predictions with probability >= 0.9
p_thresholds = [min(median, 0.9) for median in estimated_median_probs]
print(p_thresholds)


# PHASE 2: generate pseudo-labels
for i, path in enumerate(cityscapes_paths):
    x = tf.cast(read_image(path), tf.float32) / 127.5 - 1.0
    x = tf.expand_dims(x, axis=0)  # (1, 512, 1024, 3)
    
    p_pred1 = M_net1(x, training=False)  # (1, 512, 1024, 19)
    p_pred2 = M_net2(x, training=False)
    p_pred3 = M_net3(x, training=False)
    p_pred1 = tf.nn.softmax(p_pred1, axis=-1)
    p_pred2 = tf.nn.softmax(p_pred2, axis=-1)
    p_pred3 = tf.nn.softmax(p_pred3, axis=-1)
    p_pred = w1*p_pred1 + w2*p_pred2 + w3*p_pred3
    p_pred = p_pred[0]

    p_max = tf.reduce_max(p_pred, axis=-1)
    p_argmax = tf.math.argmax(p_pred, axis=-1, output_type=tf.int32)  # (512, 1024)

    for class_id in range(num_classes):
        uncertain_pixels = tf.math.logical_and(p_argmax == class_id, p_max < p_thresholds[class_id])
        p_argmax = tf.where(uncertain_pixels, 19, p_argmax)  # set some pixels to ignore_label
    
    save_path = f'{path.replace("images", "pseudo_maps").replace("jpg", "png")}'
    pseudo_map = tf.expand_dims(tf.cast(p_argmax, tf.uint8), axis=-1)
    tf.io.write_file(save_path, tf.image.encode_png(pseudo_map))

    if (i + 1) % 100 == 0:
        print(f'[{i+1}/{len(cityscapes_paths)}]')
        
