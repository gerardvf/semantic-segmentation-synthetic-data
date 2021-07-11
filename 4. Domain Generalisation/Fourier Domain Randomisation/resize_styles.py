import os
import glob
import tensorflow as tf


def read_image(path):  # read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # uint8
    return image  # (512, 1024, 3)


data_path = './raid_storage/data'
paths = sorted(glob.glob(f'{data_path}/styles/*.jpg'))

for i, path in enumerate(paths):
    image = read_image(path)
    image = tf.cast(tf.image.resize(image, [512, 1024]), tf.uint8)
    tf.io.write_file(path.replace('styles', 'resized_styles'), tf.image.encode_jpeg(image, quality=100, chroma_downsampling=False))

    if (i + 1) % 100 == 0:
        print(f'[{i+1}/{len(paths)}]')
