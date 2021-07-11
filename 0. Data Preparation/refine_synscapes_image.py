import os
import glob
import tensorflow as tf


image_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'synscapes', 'Synscapes', 'img', 'rgb', '*.png')))

for i, path in enumerate(image_paths):
    image = tf.image.decode_png(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, (512, 1024), method='bilinear')
    image = tf.cast(image, tf.uint8)
    tf.io.write_file(f'synscapes_processed/images/{str(i+1).zfill(5)}.jpg', tf.image.encode_jpeg(image))
    
    if (i + 1) % 100 == 0:
        print(f'{i + 1}/{len(image_paths)}')
