import tensorflow as tf

from models import ERFNet
from datasets import Dataset


def compute_test_iou_real(test_batch_size):  # mean and per-class IoU on testing set
    sum_intersections, sum_unions = [0] * cityscapes.num_classes, [0] * cityscapes.num_classes  # for each class id, store the sum of intersections and unions in the batch
    
    num_test_batches_per_epoch = len(cityscapes.test_paths) // test_batch_size
    for batch_id in range(num_test_batches_per_epoch):
        x, y_true = cityscapes.get_testing_batch(batch_id, test_batch_size)  # (bs, 512, 1024, 3), (bs, 512, 1024)
        p_pred = M_net(x, training=False)  # (bs, 512, 1024, 19)
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


def compute_test_iou_synth(test_batch_size):  # mean and per-class IoU on testing set
    sum_intersections, sum_unions = [0] * synscapes.num_classes, [0] * synscapes.num_classes  # for each class id, store the sum of intersections and unions in the batch
    
    num_test_batches_per_epoch = len(synscapes.test_paths) // test_batch_size
    for batch_id in range(num_test_batches_per_epoch):
        x, y_true = synscapes.get_testing_batch(batch_id, test_batch_size)  # (bs, 512, 1024, 3), (bs, 512, 1024)
        p_pred = M_net(x, training=False)  # (bs, 512, 1024, 19)
        y_pred = tf.math.argmax(p_pred, axis=-1, output_type=tf.int32)  # (bs, 512, 1024)

        for class_id in range(synscapes.num_classes):  # ignore class 'null'
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
synscapes = Dataset(name='synscapes')
M_net = ERFNet(num_classes=cityscapes.num_classes)
_ = M_net(tf.random.normal([1, 512, 1024, 3]), training=False)


models_real_test = [
    './raid_storage/weights/w86.h5',
    './raid_storage/weightssyn/w17.h5',
    './raid_storage/weightssyn/w10.h5',
    './raid_storage/weightsft20/w24.h5',
    './raid_storage/weightsft5/w52.h5',
    './raid_storage/weightsft1/w56.h5',
]

for path in models_real_test:
    M_net.load_weights(path)
    iou_mean, iou_classes = compute_test_iou_real(test_batch_size=4)
    print(path)
    print(f'Real test: {iou_mean:.4f}')
    print(' '.join([f'{score:.3f}' for score in iou_classes]))


models_synth_test = [
    './raid_storage/weightssyn/w17.h5',
]

for path in models_synth_test:
    M_net.load_weights(path)
    iou_mean, iou_classes = compute_test_iou_synth(test_batch_size=4)
    print(path)
    print(f'Synth test: {iou_mean:.4f}')
    print(' '.join([f'{score:.3f}' for score in iou_classes]))
