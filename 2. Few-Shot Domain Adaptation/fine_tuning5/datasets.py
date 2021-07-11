import glob
import random
import tensorflow as tf


def read_image(path):  # read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # uint8
    return image  # (512, 1024, 3)


def read_map(path):  # read semantic map (class id for each pixel)
    map = tf.io.read_file(path)
    map = tf.image.decode_png(map, channels=1)  # uint8
    map = tf.cast(tf.squeeze(map), tf.int32)
    return map  # (512, 1024)


# Class for Synscapes/Cityscapes dataset
class Dataset:
    def __init__(self, name):
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
                            'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
                            'motorcycle', 'bicycle', 'null']  # class 'null' (id=19) is ignored in training and not predicted
        self.name2id = {name: id for id, name in enumerate(self.class_names)}
        self.class_colors = tf.constant([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), 
                                         (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), 
                                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), 
                                         (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), 
                                         (0, 0, 230), (119, 11, 32), (0, 0, 0)], tf.uint8)
        
        self.ignore_id = 19
        self.num_classes = 19  # 20, but last one is not predicted, only in ground-truth

        data_path = './raid_storage/data'

        if name == 'synscapes':
            self.paths = list(zip(  # pairs of (image_path, map_path)
                sorted(glob.glob(f'{data_path}/synscapes/images/*.jpg')),
                sorted(glob.glob(f'{data_path}/synscapes/maps/*.png'))
            ))
            with open(f'{data_path}/synscapes_splits.txt') as f:
                lines = f.readlines()
            self.train_paths = [self.paths[int(id)] for id in lines[0].strip().split()[1:]]
            self.val_paths = [self.paths[int(id)] for id in lines[1].strip().split()[1:]]
            self.test_paths = [self.paths[int(id)] for id in lines[2].strip().split()[1:]]
        
        elif name == 'cityscapes':
            self.paths = list(zip(  # pairs of (image_path, map_path)
                sorted(glob.glob(f'{data_path}/cityscapes/images/*.jpg')),
                sorted(glob.glob(f'{data_path}/cityscapes/maps/*.png'))
            ))
            with open(f'{data_path}/cityscapes_splits.txt') as f:
                lines = f.readlines()
            self.train_paths = [self.paths[int(id)] for id in lines[0].strip().split()[1:]]
            self.val_paths = [self.paths[int(id)] for id in lines[1].strip().split()[1:]]
            self.test_paths = [self.paths[int(id)] for id in lines[2].strip().split()[1:]]
            self.train_paths = self.train_paths[:125]


    def shuffle_training_paths(self):
        self.train_paths = random.sample(self.train_paths, len(self.train_paths))

    
    def get_training_batch(self, batch_id, batch_size):
        images, maps = [], []

        for i in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image_path, map_path = self.train_paths[i]
            image, map = read_image(image_path), read_map(map_path)

            if tf.random.uniform([]) > 0.5:  # data augmentation: random horizontal flip
                image, map = tf.reverse(image, axis=[1]), tf.reverse(map, axis=[1])

            image = tf.cast(image, tf.float32) / 127.5 - 1.0  # map from range [0, 255] to [-1, 1]

            images.append(image)
            maps.append(map)

        x = tf.stack(images, axis=0)  # (bs, 512, 1024, 3) float32
        y_true = tf.stack(maps, axis=0)  # (bs, 512, 1024) int32
        return x, y_true


    def get_validation_batch(self, batch_id, batch_size):  # only for cityscapes
        images, maps = [], []

        for i in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image_path, map_path = self.val_paths[i]
            image, map = read_image(image_path), read_map(map_path)
            
            image = tf.cast(image, tf.float32) / 127.5 - 1.0

            images.append(image)
            maps.append(map)

        x = tf.stack(images, axis=0)  # (bs, 512, 1024, 3)
        y_true = tf.stack(maps, axis=0)  # (bs, 512, 1024)
        return x, y_true


    def get_testing_batch(self, batch_id, batch_size):  # only for cityscapes
        images, maps = [], []

        for i in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image_path, map_path = self.test_paths[i]
            image, map = read_image(image_path), read_map(map_path)
            
            image = tf.cast(image, tf.float32) / 127.5 - 1.0

            images.append(image)
            maps.append(map)

        x = tf.stack(images, axis=0)  # (bs, 512, 1024, 3)
        y_true = tf.stack(maps, axis=0)  # (bs, 512, 1024)
        return x, y_true
