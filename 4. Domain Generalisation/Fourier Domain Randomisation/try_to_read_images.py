import os
import glob
import tensorflow as tf


def read_image(path):  # read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # uint8
    return image  # (512, 1024, 3)


data_path = './raid_storage/data'
# paths = sorted(glob.glob(f'{data_path}/synscapes/images/*.jpg')) + sorted(glob.glob(f'{data_path}/styles/*.jpg'))
# paths = sorted(glob.glob(f'{data_path}/styles/*.jpg'))
paths = sorted(glob.glob(f'{data_path}/synscapes/images/*.jpg')) + sorted(glob.glob(f'{data_path}/resized_styles/*.jpg'))

for i, path in enumerate(paths):
    print(path)
    try:
        image = read_image(path)
    except:
        print('Error ---------------------------------------------')

    if (i + 1) % 100 == 0:
        print(f'[{i+1}/{len(paths)}]')

