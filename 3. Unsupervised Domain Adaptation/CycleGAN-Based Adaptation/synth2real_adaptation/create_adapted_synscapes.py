import os
import glob
import tensorflow as tf

from models import Generator


def read_image(path):  # read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # uint8
    return image  # (512, 1024, 3)


def save_image(image, path):  # save an image whose intensities are in range [-1, 1]
    image = (image + 1.0) * 127.5
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, tf.uint8)
    tf.io.write_file(path, tf.image.encode_jpeg(image))



G_net = Generator()
_ = G_net(tf.random.normal([1, 256, 256, 3]))
G_net.load_weights('./raid_storage/Gw212000.h5')


# translate all images
data_path = './raid_storage/data'
synscapes_paths = sorted(glob.glob(f'{data_path}/synscapes/images/*.jpg'))  # original synscapes


for i, path in enumerate(synscapes_paths):
    image = tf.cast(read_image(path), tf.float32) / 127.5 - 1.0
    image = tf.expand_dims(image, axis=0)
    save_path = f'{path.replace("synscapes", "synscapes_adapted")}'
    save_image(G_net(image)[0], save_path)

    if (i + 1) % 100 == 0:
        print(f'[{i+1}/{len(synscapes_paths)}]')
