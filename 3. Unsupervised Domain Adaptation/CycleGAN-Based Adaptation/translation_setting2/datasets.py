import glob
import random
import tensorflow as tf



def read_image(path):  # read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # uint8
    return image  # (512, 1024, 3)


def save_image(image, path):  # save an image whose intensities are in range [-1, 1]
    image = (image + 1.0) * 127.5
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, tf.uint8)
    tf.io.write_file(path, tf.image.encode_jpeg(image))


class Dataset:
    def __init__(self):
        data_path = './raid_storage/data'
        
        self.source_paths = sorted(glob.glob(f'{data_path}/synscapes/images/*.jpg'))  # source domain
        with open(f'{data_path}/synscapes_splits.txt') as f:
            lines = f.readlines()
        self.source_paths = [self.source_paths[int(id)] for id in lines[0].strip().split()[1:]]  # only use training set

        self.target_paths = sorted(glob.glob(f'{data_path}/cityscapes/images/*.jpg'))  # target domain
        with open(f'{data_path}/cityscapes_splits.txt') as f:
            lines = f.readlines()
        self.target_paths = [self.target_paths[int(id)] for id in lines[0].strip().split()[1:]]  # only use training set


    def sample_batch(self, crop_size):  # batch_size=1, i.e. return a random source image and a random target image
        source_image = read_image(random.choice(self.source_paths))
        source_image = tf.image.random_crop(source_image, size=[*crop_size, 3])
        source_image = tf.image.random_flip_left_right(source_image)  # random horizontal flip
        source_image = tf.cast(source_image, tf.float32) / 127.5 - 1.0  # map from range [0, 255] to [-1, 1]
        
        target_image = read_image(random.choice(self.target_paths))
        target_image = tf.image.random_crop(target_image, size=[*crop_size, 3])
        target_image = tf.image.random_flip_left_right(target_image)
        target_image = tf.cast(target_image, tf.float32) / 127.5 - 1.0

        return tf.expand_dims(source_image, axis=0), tf.expand_dims(target_image, axis=0)  # (1, H, W, 3) float32, (1, H, W, 3) float32


class FakeImagePool:
    # store history of fake source and target images
    def __init__(self):
        self.max_size = 50
        self.fake_images = []

    def query(self, fake_image):  # given a generated fake image, either return it or an image from the pool (and update the pool accordingly)
        if len(self.fake_images) < self.max_size:
            self.fake_images.append(fake_image)
            return fake_image
        
        if random.random() > 0.5:
            return fake_image
        else:
            random_id = random.randint(0, self.max_size - 1)
            old_fake_image = self.fake_images[random_id]
            self.fake_images[random_id] = fake_image
            return old_fake_image
