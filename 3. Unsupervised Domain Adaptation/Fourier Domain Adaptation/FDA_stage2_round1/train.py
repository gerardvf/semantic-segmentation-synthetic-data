import os
import tensorflow as tf

from models import ERFNet
from datasets import Dataset, create_fourier_mask


def cross_entropy_loss(y_true, p_pred):  # (bs, 512, 1024), (bs, 512, 1024, 19) with no softmax applied, i.e. logits. note 19 non-null predicted classes
    y_true = tf.reshape(y_true, [-1])
    p_pred = tf.reshape(p_pred, [-1, cityscapes.num_classes])
    indices = tf.squeeze(tf.where(tf.not_equal(y_true, cityscapes.ignore_id)))  # indices to keep i.e. class label not 19
    y_true = tf.gather(y_true, indices)
    p_pred = tf.gather(p_pred, indices)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_pred, labels=y_true))
    return loss


def compute_val_iou_city(val_batch_size):  # mean IoU on validation set (val_batch_size should be a divisor of the number of val images)
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



synscapes = Dataset(name='synscapes')
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


lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([15000, 30000], [4e-4, 2e-4, 1e-4])
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

weights_path = './raid_storage/weightsfda3-s2'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print(f'{weights_path} directory created.')


# Training
num_epochs = 20
batch_size = 2  # 2 synthetic adapted images + 2 real images with pseudolabels
num_batches_per_epoch = len(synscapes.train_paths) // batch_size

beta = 0.18  # 0.02, 0.1, 0.18
fourier_mask = create_fourier_mask([512, 1024, 3], beta)

# def save_image(image, path):  # save an image whose intensities are in range [-1, 1]
#     image = (image + 1.0) * 127.5
#     image = tf.clip_by_value(image, 0.0, 255.0)
#     image = tf.cast(image, tf.uint8)
#     tf.io.write_file(path, tf.image.encode_jpeg(image))

# def save_map(m, path):  # save an image whose intensities are in range [-1, 1]
#     class_colors = tf.constant([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
#                             [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
#                             [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], 
#                             [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], 
#                             [0, 0, 230], [119, 11, 32], [0, 0, 0]], tf.uint8)
#     tf.io.write_file(path, tf.image.encode_png(tf.gather(class_colors, m)))

iou_mean_city = compute_val_iou_city(val_batch_size=4)
print(f'[Epoch 0/{num_epochs}]. Validation mIoU: {iou_mean_city:.4f}')
for epoch_id in range(num_epochs):
    synscapes.shuffle_training_paths()
    for batch_id in range(num_batches_per_epoch):
        x, y_true = synscapes.get_training_batch(batch_id, batch_size, fourier_mask, cityscapes.train_paths)
        
        # save_image(x[0], 'b1.jpg')
        # save_image(x[1], 'b2.jpg')
        # save_image(x[2], 'b3.jpg')
        # save_image(x[3], 'b4.jpg')
        # save_map(y_true[0], 'b1l.png')
        # save_map(y_true[1], 'b2l.png')
        # save_map(y_true[2], 'b3l.png')
        # save_map(y_true[3], 'b4l.png')
        
        loss = train_step(x, y_true)

        if (batch_id + 1) % 500 == 0:
            print(f'[Epoch {epoch_id+1}/{num_epochs}. Batch {batch_id+1}/{num_batches_per_epoch}] Training batch loss: {loss:.4f}')

    # Compute the IoU score on validation set (every epoch)
    iou_mean_city = compute_val_iou_city(val_batch_size=4)
    print(f'[Epoch {epoch_id+1}/{num_epochs}]. Validation mIoU: {iou_mean_city:.4f}')

    # Save current weights as a h5 file (every epoch)
    h5_save_path = f'{weights_path}/w{epoch_id+1}.h5'
    M_net.save_weights(h5_save_path)
    print(f'Weights saved at {h5_save_path}.')
