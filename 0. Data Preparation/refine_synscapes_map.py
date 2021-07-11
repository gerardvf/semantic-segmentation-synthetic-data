import os
import glob
import tensorflow as tf


# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py.
# note: last class is not used in training or testing

class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
                'motorcycle', 'bicycle', 'other']  # class 'other' includes the rest of classes

class_colors = tf.constant([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
                            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
                            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], 
                            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], 
                            [0, 0, 230], [119, 11, 32], [0, 0, 0]], tf.uint8)

# Used to read annotation files, which contain ids (0 to 34) instead of class labels (0 to 19).
id2label = tf.constant([19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19, 2, 3, 
                        4, 19, 19, 19, 5, 19, 6, 7, 8, 9, 10, 11, 12, 
                        13, 14, 15, 19, 19, 16, 17, 18, 19], tf.uint8)


labels_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'synscapes', 'Synscapes', 'img', 'class', '*.png')))

for i, path in enumerate(labels_paths):
    labels = tf.image.decode_png(tf.io.read_file(path), channels=1)
    labels = tf.image.resize(labels, (512, 1024), method='nearest')
    labels = tf.gather(id2label, tf.cast(labels, tf.int32))
    tf.io.write_file(f'synscapes_processed/labels/{str(i+1).zfill(5)}.png', tf.image.encode_png(labels))

    if (i + 1) % 100 == 0:
        print(f'{i + 1}/{len(labels_paths)}')


# # FOR LATER

# # Load labels
# path = 'class10r.png'
# labels = tf.image.decode_png(tf.io.read_file(path), channels=1)
# labels = tf.cast(tf.squeeze(labels), tf.int32)

# # Save colored labels
# colored_labels = tf.gather(class_colors, labels)
# tf.io.write_file('class10rc.png', tf.image.encode_png(colored_labels))

# # note: uint8 to save images, int32 for indices
