import tensorflow as tf

from models import Generator, ERFNet
from datasets import Dataset

def save_image(image, path):  # save an image whose intensities are in range [-1, 1]
    image = (image + 1.0) * 127.5
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, tf.uint8)
    tf.io.write_file(path, tf.image.encode_jpeg(image))

def compute_test_iou_real(test_batch_size):  # mean and per-class IoU on testing set
    sum_intersections, sum_unions = [0] * cityscapes.num_classes, [0] * cityscapes.num_classes  # for each class id, store the sum of intersections and unions in the batch
    
    num_test_batches_per_epoch = len(cityscapes.test_paths) // test_batch_size
    for batch_id in range(num_test_batches_per_epoch):
        x, y_true = cityscapes.get_testing_batch(batch_id, test_batch_size)  # (bs, 512, 1024, 3), (bs, 512, 1024)
        xtranslated = F_net(x)
        p_pred = M_net(xtranslated, training=False)  # (bs, 512, 1024, 19)
        y_pred = tf.math.argmax(p_pred, axis=-1, output_type=tf.int32)  # (bs, 512, 1024)

        for class_id in range(cityscapes.num_classes):  # ignore class 'null'
            true_equal_class = tf.cast(tf.equal(y_true, class_id), tf.int32)
            pred_equal_class = tf.cast(tf.equal(y_pred, class_id), tf.int32)

            batch_intersection = tf.reduce_sum(tf.multiply(true_equal_class, pred_equal_class))  # TP (true positives)
            batch_union = tf.reduce_sum(true_equal_class) + tf.reduce_sum(pred_equal_class) - batch_intersection  # TP + FP + FN = (TP + FP) + (TP + FN) - TP
            
            sum_intersections[class_id] += batch_intersection
            sum_unions[class_id] += batch_union

    iou_classes = tf.divide(sum_intersections, sum_unions)
    iou_mean = tf.reduce_mean(iou_classes)
    return iou_mean, iou_classes


cityscapes = Dataset(name='cityscapes')

M_net = ERFNet(num_classes=cityscapes.num_classes)
_ = M_net(tf.random.normal([1, 512, 1024, 3]), training=False)
M_net.load_weights('w17.h5')

F_net = Generator()
_ = F_net(tf.random.normal([1, 256, 256, 3]))
F_net.load_weights('Fw212000.h5')

iou_mean, iou_classes = compute_test_iou_real(test_batch_size=4)
print(f'Real test: {iou_mean:.4f}')
print(' '.join([f'{score:.3f}' for score in iou_classes]))
