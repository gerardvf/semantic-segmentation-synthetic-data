import os
import tensorflow as tf

from models import ERFNet
from datasets import Dataset


def cross_entropy_loss(y_true, p_pred):  # (bs, 512, 1024), (bs, 512, 1024, 19) with no softmax applied, i.e. logits. note 19 non-null predicted classes
    y_true = tf.reshape(y_true, [-1])
    p_pred = tf.reshape(p_pred, [-1, cityscapes.num_classes])
    indices = tf.squeeze(tf.where(tf.not_equal(y_true, cityscapes.ignore_id)))  # indices to keep i.e. class label not 19
    y_true = tf.gather(y_true, indices)
    p_pred = tf.gather(p_pred, indices)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_pred, labels=y_true))
    return loss


def compute_val_iou(val_batch_size):  # mean IoU on validation set (val_batch_size should be a divisor of the number of val images)
    sum_intersections, sum_unions = [0] * cityscapes.num_classes, [0] * cityscapes.num_classes  # for each class id, store the sum of intersections and unions in the batch (unit: number of pixels)
    
    num_val_batches_per_epoch = len(cityscapes.val_paths) // val_batch_size
    for batch_id in range(num_val_batches_per_epoch):
        x, y_true = cityscapes.get_validation_batch(batch_id, val_batch_size)  # (bs, 512, 1024, 3), (bs, 512, 1024)
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
    return iou_mean



cityscapes = Dataset(name='cityscapes')
M_net = ERFNet(num_classes=cityscapes.num_classes)
_ = M_net(tf.random.normal([1, 512, 1024, 3]), training=False)


@tf.function
def train_step(x, y_true):
    print('Tracing training step...')

    # Forward pass
    with tf.GradientTape() as tape:
        p_pred = M_net(x)  # (bs, 512, 1024, 19)
        loss = cross_entropy_loss(y_true, p_pred)

    # Backward pass
    grads = tape.gradient(loss, M_net.trainable_variables)
    opt.apply_gradients(zip(grads, M_net.trainable_variables))

    return loss


opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

weights_path = './raid_storage/weightsreal5'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print(f'{weights_path} directory created.')


# Training
num_epochs = 700
batch_size = 4
num_batches_per_epoch = len(cityscapes.train_paths) // batch_size

# iou_mean = compute_val_iou(val_batch_size=4)
# print(f'[Epoch 0/{num_epochs}]. Validation mIoU: {iou_mean:.4f}')
for epoch_id in range(num_epochs):
    cityscapes.shuffle_training_paths()
    for batch_id in range(num_batches_per_epoch):
        x, y_true = cityscapes.get_training_batch(batch_id, batch_size)
        loss = train_step(x, y_true)
 
        if (batch_id + 1) % 20 == 0:
            print(f'[Epoch {epoch_id+1}/{num_epochs}. Batch {batch_id+1}/{num_batches_per_epoch}] Training batch loss: {loss:.4f}')

    if (epoch_id + 1) % 100 == 0:
        # Compute the IoU score on validation set (every epoch)
        iou_mean = compute_val_iou(val_batch_size=4)
        print(f'[Epoch {epoch_id+1}/{num_epochs}]. Validation mIoU: {iou_mean:.4f}')

        # Save current weights as a h5 file (every epoch)
        h5_save_path = f'{weights_path}/w{epoch_id+1}.h5'
        M_net.save_weights(h5_save_path)
        print(f'Weights saved at {h5_save_path}.')
