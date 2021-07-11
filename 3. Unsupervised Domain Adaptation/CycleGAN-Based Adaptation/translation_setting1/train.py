import os
import tensorflow as tf

from models import Generator, Discriminator
from datasets import Dataset, FakeImagePool, save_image



weights_path = './raid_storage/ganweights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print(f'{weights_path} directory created.')

samples_path = './raid_storage/gansamples'
if not os.path.exists(samples_path):
    os.makedirs(samples_path)
    print(f'{samples_path} directory created.')


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # Learning rate: 2e-4 during the first 100k steps, linearly decayed to 0 during the second 100k steps
    def __init__(self, initial_lr, total_steps, step_decay):
        super(CustomSchedule, self).__init__()
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.step_decay = tf.cast(step_decay, tf.float32)

    def __call__(self, step):
        return tf.where(tf.less(step, self.step_decay),
                        self.initial_lr,
                        self.initial_lr * (1 - (step - self.step_decay) / (self.total_steps - self.step_decay)))


H, W = 452, 452  # height and width of cropped images
LAMBDA_CYC = 10.0
NUM_STEPS = 200000

dataset = Dataset()

G_net = Generator()
F_net = Generator()
E_net = Discriminator()
H_net = Discriminator()

_ = G_net(tf.random.normal([1, H, W, 3]))
_ = F_net(tf.random.normal([1, H, W, 3]))
_ = E_net(tf.random.normal([1, H, W, 3]))
_ = H_net(tf.random.normal([1, H, W, 3]))

lr_schedule = CustomSchedule(initial_lr=2e-4, total_steps=200000, step_decay=100000)
GF_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
EH_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)


@tf.function
def GF_train_step(real_source, real_target):  # (1, H, W, 3), (1, H, W, 3)
    print('Tracing GF_train_step...')

    # Forward pass
    with tf.GradientTape() as tape:
        fake_target = G_net(real_source)
        fake_source = F_net(real_target)
        reconstructed_target = G_net(fake_source)
        reconstructed_source = F_net(fake_target)
        # note: no identity loss?
        p_fake_source = E_net(fake_source)
        p_fake_target = H_net(fake_target)

        G_adv_loss = tf.reduce_mean(tf.math.squared_difference(p_fake_target, 1))
        F_adv_loss = tf.reduce_mean(tf.math.squared_difference(p_fake_source, 1))
        GF_cyc_loss = LAMBDA_CYC * (tf.reduce_mean(tf.math.abs(real_source - reconstructed_source)) + 
                                    tf.reduce_mean(tf.math.abs(real_target - reconstructed_target)))
        GF_total_loss = G_adv_loss + F_adv_loss + GF_cyc_loss
        
    # Backward pass
    GF_grads = tape.gradient(GF_total_loss, G_net.trainable_variables + F_net.trainable_variables)
    GF_opt.apply_gradients(zip(GF_grads, G_net.trainable_variables + F_net.trainable_variables))

    return GF_total_loss, GF_cyc_loss, fake_source, fake_target


@tf.function
def EH_train_step(real_source, real_target, fake_source, fake_target):  # (1, H, W, 3), (1, H, W, 3), (1, H, W, 3), (1, H, W, 3)
    print('Tracing EH_train_step...')

    # Forward pass
    with tf.GradientTape() as tape:
        p_real_source = E_net(real_source)
        p_real_target = H_net(real_target)
        p_fake_source = E_net(fake_source)
        p_fake_target = H_net(fake_target)

        E_adv_loss = (tf.reduce_mean(tf.math.squared_difference(p_real_source, 1)) +
                      tf.reduce_mean(tf.math.squared_difference(p_fake_source, 0)))
        H_adv_loss = (tf.reduce_mean(tf.math.squared_difference(p_real_target, 1)) +
                      tf.reduce_mean(tf.math.squared_difference(p_fake_target, 0)))
        EH_total_loss = 0.5 * (E_adv_loss + H_adv_loss)

    # Backward pass
    EH_grads = tape.gradient(EH_total_loss, E_net.trainable_variables + H_net.trainable_variables)
    EH_opt.apply_gradients(zip(EH_grads, E_net.trainable_variables + H_net.trainable_variables))

    return EH_total_loss


source_pool = FakeImagePool()
target_pool = FakeImagePool()

for step_id in range(NUM_STEPS):
    real_source, real_target = dataset.sample_batch(crop_size=[H, W])
    GF_total_loss, GF_cyc_loss, fake_source, fake_target = GF_train_step(real_source, real_target)
    
    if (step_id + 1) % 2000 == 0:  # Save current weights of generators as a h5 file, and also real_source and fake_target (uploaded to the cloud too)
        
        G_save_path = f'{weights_path}/Gw{step_id+1}.h5'
        G_net.save_weights(G_save_path)
        print(f'Weights of G saved at {G_save_path}.')
        F_save_path = f'{weights_path}/Fw{step_id+1}.h5'
        F_net.save_weights(F_save_path)
        print(f'Weights of F saved at {F_save_path}.')
        
        real_source_save_path = f'{samples_path}/{step_id+1}s.jpg'
        save_image(real_source[0], real_source_save_path)
        # with open(real_source_save_path, 'rb') as f:
        #     dbx.files_upload(f.read(), f'/cyclegan/{step_id+1}s.jpg')

        translated_source_save_path = f'{samples_path}/{step_id+1}st.jpg'
        save_image(fake_target[0], translated_source_save_path)
        # with open(translated_source_save_path, 'rb') as f:
        #     dbx.files_upload(f.read(), f'/cyclegan/{step_id+1}st.jpg')

    fake_source = source_pool.query(fake_source)
    fake_target = target_pool.query(fake_target)
    EH_total_loss = EH_train_step(real_source, real_target, fake_source, fake_target)

    if (step_id + 1) % 200 == 0:
        print(f'[Step {step_id+1}/{NUM_STEPS}] GF_total_loss: {GF_total_loss:.4f}. GF_cyc_loss: {GF_cyc_loss:.4f}. EH_total_loss: {EH_total_loss:.4f}')
        print(f'Learning rates: {GF_opt.learning_rate(step_id):.2e} {EH_opt.learning_rate(step_id):.2e}')
