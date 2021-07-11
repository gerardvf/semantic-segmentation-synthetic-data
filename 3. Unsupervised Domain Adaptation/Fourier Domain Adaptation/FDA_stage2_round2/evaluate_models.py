import tensorflow as tf

from models import ERFNet
from datasets import Dataset


def compute_test_iou_real(net, test_batch_size):  # mean and per-class IoU on testing set
    sum_intersections, sum_unions = [0] * cityscapes.num_classes, [0] * cityscapes.num_classes  # for each class id, store the sum of intersections and unions in the batch
    
    num_test_batches_per_epoch = len(cityscapes.test_paths) // test_batch_size
    for batch_id in range(num_test_batches_per_epoch):
        x, y_true = cityscapes.get_testing_batch(batch_id, test_batch_size)  # (bs, 512, 1024, 3), (bs, 512, 1024)
        p_pred = net(x, training=False)  # (bs, 512, 1024, 19)
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



def compute_test_iou_real_ensemble(net1, net2, net3, test_batch_size):  # mean and per-class IoU on testing set
    sum_intersections, sum_unions = [0] * cityscapes.num_classes, [0] * cityscapes.num_classes  # for each class id, store the sum of intersections and unions in the batch
    
    w1, w2, w3 = 0.5, 0.2, 0.3
    num_test_batches_per_epoch = len(cityscapes.test_paths) // test_batch_size
    for batch_id in range(num_test_batches_per_epoch):
        x, y_true = cityscapes.get_testing_batch(batch_id, test_batch_size)  # (bs, 512, 1024, 3), (bs, 512, 1024)
        p_pred1 = net1(x, training=False)  # (bs, 512, 1024, 19)
        p_pred2 = net2(x, training=False)
        p_pred3 = net3(x, training=False)
        p_pred1 = tf.nn.softmax(p_pred1, axis=-1)
        p_pred2 = tf.nn.softmax(p_pred2, axis=-1)
        p_pred3 = tf.nn.softmax(p_pred3, axis=-1)
        p_pred = w1*p_pred1 + w2*p_pred2 + w3*p_pred3
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
M_net1 = ERFNet(num_classes=cityscapes.num_classes)
M_net2 = ERFNet(num_classes=cityscapes.num_classes)
M_net3 = ERFNet(num_classes=cityscapes.num_classes)
_ = M_net1(tf.random.normal([1, 512, 1024, 3]), training=False)
_ = M_net2(tf.random.normal([1, 512, 1024, 3]), training=False)
_ = M_net3(tf.random.normal([1, 512, 1024, 3]), training=False)
M_net1.load_weights('./raid_storage/weightsfda1-s3/w17.h5')
M_net2.load_weights('./raid_storage/weightsfda2-s3/w14.h5')
M_net3.load_weights('./raid_storage/weightsfda3-s3/w14.h5')

#iou_mean, iou_classes = compute_test_iou_real(M_net1, test_batch_size=4)
#print(f'Real test: {iou_mean:.4f}')
#print(' '.join([f'{score:.3f}' for score in iou_classes]))

#iou_mean, iou_classes = compute_test_iou_real(M_net2, test_batch_size=4)
#print(f'Real test: {iou_mean:.4f}')
#print(' '.join([f'{score:.3f}' for score in iou_classes]))

#iou_mean, iou_classes = compute_test_iou_real(M_net3, test_batch_size=4)
#print(f'Real test: {iou_mean:.4f}')
#print(' '.join([f'{score:.3f}' for score in iou_classes]))

iou_mean, iou_classes = compute_test_iou_real_ensemble(M_net1, M_net2, M_net3, test_batch_size=4)
print(f'Real test: {iou_mean:.4f}')
print(' '.join([f'{score:.3f}' for score in iou_classes]))



