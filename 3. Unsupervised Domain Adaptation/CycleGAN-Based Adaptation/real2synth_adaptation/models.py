import tensorflow as tf
import tensorflow_addons as tfa



class ResBlock(tf.keras.layers.Layer):
    def __init__(self, ch_out):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 3), strides=1, padding='valid', use_bias=False)
        self.in1 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 3), strides=1, padding='valid', use_bias=False)
        self.in2 = tfa.layers.InstanceNormalization(epsilon=1e-5)

    def call(self, input):
        x = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = tf.nn.relu(self.in1(self.conv1(x)))
        
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = tf.nn.relu(self.in2(self.conv2(x)))
        
        x = tf.keras.layers.add([x, input])
        return x

    def plot(self):
        input = tf.keras.Input(shape=[64, 64, 256])
        model = tf.keras.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(model, to_file='ResBlock.png', show_shapes=True, show_layer_names=False)
        

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=1, padding='valid', use_bias=False)
        self.in1 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in2 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in3 = tfa.layers.InstanceNormalization(epsilon=1e-5)

        self.resblocks = [
            ResBlock(ch_out=256), ResBlock(ch_out=256), ResBlock(ch_out=256), 
            ResBlock(ch_out=256), ResBlock(ch_out=256), ResBlock(ch_out=256), 
            ResBlock(ch_out=256), ResBlock(ch_out=256), ResBlock(ch_out=256), 
        ]

        self.conv4 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in4 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv5 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', use_bias=False)
        self.in5 = tfa.layers.InstanceNormalization(epsilon=1e-5)
        self.conv6 = tf.keras.layers.Conv2D(3, kernel_size=(7, 7), strides=1, padding='valid', use_bias=False)

    def call(self, input):
        x = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = tf.nn.relu(self.in1(self.conv1(x)))

        x = tf.nn.relu(self.in2(self.conv2(x)))
        
        x = tf.nn.relu(self.in3(self.conv3(x)))

        for block in self.resblocks:
            x = block(x)

        x = tf.nn.relu(self.in4(self.conv4(x)))

        x = tf.nn.relu(self.in5(self.conv5(x)))

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.conv6(x)  # tanh instead of instancenorm and relu

        x = tf.tanh(x)  # to range [-1, 1], so that then it can be transformed to the range [0, 255]
        return x

    def plot(self):
        input = tf.keras.Input(shape=[256, 256, 3])
        model = tf.keras.Model(inputs=input, outputs=self.call(input))
        tf.keras.utils.plot_model(model, to_file='Generator.png', show_shapes=True, show_layer_names=False)


class DownBlock(tf.keras.layers.Layer):
    def __init__(self, ch_in, ch_out):
        super(DownBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(ch_out - ch_in, kernel_size=(3, 3), strides=2, padding='same')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, input, training=True):
        x1 = self.conv(input)
        x2 = self.pool(input)
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])
        x = self.bn(x, training=training)
        return x

    def plot(self):
        input = tf.keras.Input(shape=[256, 128, 16])
        model = tf.keras.Model(inputs=input, outputs=self.call(input, training=False))
        tf.keras.utils.plot_model(model, to_file='DownBlock.png', show_shapes=True, show_layer_names=False)


class UpBlock(tf.keras.layers.Layer):
    def __init__(self, ch_out):
        super(UpBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(ch_out, kernel_size=(3, 3), strides=2, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, input, training=True):
        x = self.conv(input)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

    def plot(self):
        input = tf.keras.Input(shape=[256, 128, 16])
        model = tf.keras.Model(inputs=input, outputs=self.call(input, training=False))
        tf.keras.utils.plot_model(model, to_file='UpBlock.png', show_shapes=True, show_layer_names=False)


class ResBlock1D(tf.keras.layers.Layer):
    def __init__(self, ch_out, dropout_rate=0, dilation_rate=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 1), strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(ch_out, kernel_size=(1, 3), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(ch_out, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(dilation_rate, 1))
        self.conv4 = tf.keras.layers.Conv2D(ch_out, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, dilation_rate))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout_rate = dropout_rate
        if self.dropout_rate != 0:
            self.drop = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input, training=True):
        x = self.conv1(input)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x = self.bn2(x, training=training)

        if self.dropout_rate != 0:
            x = self.drop(x, training=training)
        
        x = tf.keras.layers.Add()([x, input])
        x = tf.nn.relu(x)
        return x

    def plot(self):  # input and output channels must be equal
        input = tf.keras.Input(shape=[256, 128, 16])
        model = tf.keras.Model(inputs=input, outputs=self.call(input, training=False))
        tf.keras.utils.plot_model(model, to_file='ResBlock1D.png', show_shapes=True, show_layer_names=False)


class ERFNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ERFNet, self).__init__()
        self.encoder_layers = [
            DownBlock(ch_in=3, ch_out=16),
            DownBlock(ch_in=16, ch_out=64),
            ResBlock1D(ch_out=64, dropout_rate=0.03),
            ResBlock1D(ch_out=64, dropout_rate=0.03),
            ResBlock1D(ch_out=64, dropout_rate=0.03),
            ResBlock1D(ch_out=64, dropout_rate=0.03),
            ResBlock1D(ch_out=64, dropout_rate=0.03),
            DownBlock(ch_in=64, ch_out=128),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=2),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=4),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=8),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=16),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=2),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=4),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=8),
            ResBlock1D(ch_out=128, dropout_rate=0.3, dilation_rate=16),
        ]
        self.decoder_layers = [
            UpBlock(ch_out=64),
            ResBlock1D(ch_out=64),
            ResBlock1D(ch_out=64),
            UpBlock(ch_out=16),
            ResBlock1D(ch_out=16),
            ResBlock1D(ch_out=16),
            tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(2, 2), strides=2, padding='same'),
        ]
    
    def call(self, input, training=True):
        x = input
        for layer in self.encoder_layers + self.decoder_layers:
            x = layer(x, training=training)
        return x

    def plot(self):
        input = tf.keras.Input(shape=[512, 1024, 3])
        model = tf.keras.Model(inputs=input, outputs=self.call(input, training=False))
        tf.keras.utils.plot_model(model, to_file='ERFNet.png', show_shapes=True, show_layer_names=False)
