# add block 5 and 6 to the encoder, decoder is same
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose


#########################
#        ENCODER        #
#########################

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim, concat_input_and_condition=True):

        super(Encoder, self).__init__()
        self.use_cond_input = concat_input_and_condition
        self.enc_block_1 = Conv2D(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_2 = Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_3 = Conv2D(
            filters=128,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_4 = Conv2D(
            filters=256,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_5 = Conv2D(
            filters=256*2,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.enc_block_6 = Conv2D(
            filters=256*2*2,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.flatten = tf.keras.layers.Flatten()
        #self.dense = tf.keras.layers.Dense(latent_dim + latent_dim)
        self.dense = tf.keras.layers.Dense(latent_dim + latent_dim)

    def __call__(self, input_img, input_label, conditional_input, latent_dim, is_train):
        # Encoder block 1
        # conditional_input (32, 64, 64, 44)
        # print("conditional_input", conditional_input.shape)
        # conditional_input = tf.random.uniform(shape=[32, 64, 64, 43])

        if self.use_cond_input:
            # x = conditional_input
            x = tf.keras.layers.InputLayer(input_shape=(conditional_input.shape))(conditional_input)
        else:
            # add the condition to the last conv layer before dense when the tensor size is 4 x4
            x = tf.keras.layers.InputLayer(input_shape=(input_img.shape))(input_img)
            cond = input_label  # tf.random.uniform(shape=[32, 512])
            cond = tf.reshape(cond, [input_img.shape[0], 4, 4, -1])

        #print("input to encoder", x.shape) # (1, 64, 64, 515)
        x = self.enc_block_1(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        # Encoder block 2
        x = self.enc_block_2(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        # Encoder block 3
        x = self.enc_block_3(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        # Encoder block 4
        x = self.enc_block_4(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)
        # x = (32, 4, 4, 256)

        if not self.use_cond_input:
            x = tf.concat([x, cond], axis=3)
            # print("x", x.shape) #x (32, 4, 4, 288)

        #Encoder block 5
        x = self.enc_block_5(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)
        #
        # Encoder block 6
        x = self.enc_block_6(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        #print("x before dense", x.shape) # x before dense (1, 1, 1, 1024)
        x = self.flatten(x)
        #print("x after flatten", x.shape) # x after flatten (1, 1024) #
        x = self.dense(x) #x after dense (32, 256) ori
        #print("x after dense", x.shape)

        return x


#########################
#        DECODER        #
#########################

class Decoder(tf.keras.Model):

    def __init__(self, batch_size=32):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        # 4 * 4 * 512
        # self.dense = tf.keras.layers.Dense(4*4*self.batch_size*8) #4,096#
        self.dense = tf.keras.layers.Dense(4 * 4 * 512)
        self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 512))

        # input dim * stride if padding is same
        self.dec_block_6 = Conv2DTranspose(
            filters=256*2*2,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_7 = Conv2DTranspose(
            filters=256*2,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_1 = Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_2 = Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_3 = Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_4 = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

        self.dec_block_5 = Conv2DTranspose(
            filters=3,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            kernel_initializer=he_normal())

    def __call__(self, z_cond, is_train):
        # Reshape input

        # z_cond decoder (32, 168)
        # print("z_cond decoder", z_cond.shape)
        #print("x before decoder", z_cond.shape) # x before encoder (32, 640)
        x = self.dense(z_cond)
        # x (32, 4096)
        # print("x", x.shape)
        x = tf.nn.leaky_relu(x)
        x = self.reshape(x)

        # x reshape (32, 4, 4, 256)
        # print("x reshape", x.shape)

        # Decoder block 6
        # x = self.dec_block_6(x)
        # x = BatchNormalization(trainable=is_train)(x)
        # x = tf.nn.leaky_relu(x)
        #
        # #Decoder block 7
        # x = self.dec_block_7(x)
        # x = BatchNormalization(trainable=is_train)(x)
        # x = tf.nn.leaky_relu(x)


        # Decoder block 1
        x = self.dec_block_1(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Decoder block 2
        x = self.dec_block_2(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Decoder block 3
        x = self.dec_block_3(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)
        # Decoder block 4
        x = self.dec_block_4(x)
        x = BatchNormalization(trainable=is_train)(x)
        x = tf.nn.leaky_relu(x)

        x = self.dec_block_5(x)
        #print("x out decoder", x.shape) #x out decoder (32, 64, 64, 3)
        #x = tf.random.uniform(shape=(32, 64, 64, 3))
        return x


#########################
#       Conv-CVAE       #
#########################

class ConvCVAE(tf.keras.Model):

    def __init__(self,
                 encoder,
                 decoder,
                 label_dim,
                 latent_dim,
                 batch_size=32,
                 beta=1,
                 image_dim=[64, 64, 3]):
        super(ConvCVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta
        self.image_dim = image_dim = [64, 64, 3]

    def __call__(self, inputs, is_train):
        input_img, input_label, conditional_input = self.conditional_input(inputs)

        # z_mean, z_log_var = tf.split(self.encoder(conditional_input, self.latent_dim, is_train),
        # num_or_size_splits=2, axis=1)
        z_mean, z_log_var = tf.split(
            self.encoder(input_img, input_label, conditional_input, self.latent_dim, is_train), num_or_size_splits=2,
            axis=1)
        #print("....done encoding")
        z_cond = self.reparametrization(z_mean, z_log_var, input_label)
        logits = self.decoder(z_cond, is_train)
        # print("....done decoding")

        recon_img = tf.nn.sigmoid(logits)

        # Loss computation #
        latent_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                                            axis=-1)  # KL divergence
        reconstr_loss = np.prod((64, 64)) * tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(input_img),
                                                                                tf.keras.backend.flatten(
                                                                                    recon_img))  # over weighted MSE
        loss = reconstr_loss + self.beta * latent_loss  # weighted ELBO loss
        loss = tf.reduce_mean(loss)

        return {
            'recon_img': recon_img,
            'latent_loss': latent_loss,
            'reconstr_loss': reconstr_loss,
            'loss': loss,
            'z_mean': z_mean,
            'z_log_var': z_log_var
        }

    def conditional_input(self, inputs):
        """
        Builds the conditional input and returns the original input images, their labels and the conditional input.
        inputs is a tuple where inputs[0] is numpy.ndarray of shape (batch, 64, 64, 3) and
        inputs[1] has the shape (batch, label_dim).

        """

        labels_ = inputs[1]  # tf.random.uniform(shape=[inputs[0].shape[0], 41])

        input_img = tf.keras.layers.InputLayer(input_shape=self.image_dim, dtype='float32')(inputs[0])
        input_label = tf.keras.layers.InputLayer(input_shape=(self.label_dim,), dtype='float32')(labels_)
        # labels = (batch_size, 1, 1, label_size)
        labels = tf.reshape(labels_, [-1, 1, 1, self.label_dim])
        # ones = (batch_size, 64, 64, label_size)
        ones = tf.ones([inputs[0].shape[0]] + self.image_dim[0:-1] + [self.label_dim])
        labels = ones * labels  # (batch_size, 64, 64, label_size)
        conditional_input = tf.keras.layers.InputLayer(
            input_shape=(self.image_dim[0], self.image_dim[1], self.image_dim[2] + self.label_dim), dtype='float32')(
            tf.concat([inputs[0], labels], axis=3))

        # input_img = (batch_size, 64, 64, 3)
        # input_label = (batch_size, label_dim)
        # conditional_input = (batch_size, 64, 64, label_dim + 3)

        return input_img, input_label, conditional_input

    def conditional_input_ori(self, inputs):
        """ Builds the conditional input and returns the original input images, their labels and the conditional
        input."""

        # print("inputs", type(inputs)) # data <class 'tuple'>
        # print("inputs[0]", inputs[0].shape) # inputs[0](1, 64, 64, 3)
        # print(inputs[1])
        inputs[1] = tf.random.uniform(shape=[1, 41])
        # print("inputs[1]", inputs[1].shape) #inputs[1](1, 40)
        # print("inputs[0]", type(inputs[0])) <class 'numpy.ndarray'>
        input_img = tf.keras.layers.InputLayer(input_shape=self.image_dim, dtype='float32')(inputs[0])
        input_label = tf.keras.layers.InputLayer(input_shape=(self.label_dim,), dtype='float32')(inputs[1])
        labels = tf.reshape(inputs[1], [-1, 1, 1, self.label_dim])  # batch_size, 1, 1, label_size
        ones = tf.ones(
            [inputs[0].shape[0]] + self.image_dim[0:-1] + [self.label_dim])  # batch_size, 64, 64, label_size
        labels = ones * labels  # batch_size, 64, 64, label_size
        conditional_input = tf.keras.layers.InputLayer(
            input_shape=(self.image_dim[0], self.image_dim[1], self.image_dim[2] + self.label_dim), dtype='float32')(
            tf.concat([inputs[0], labels], axis=3))

        return input_img, input_label, conditional_input

    def reparametrization(self, z_mean, z_log_var, input_label):
        """ Performs the re-parametrization trick"""

        eps = tf.random.normal(shape=(input_label.shape[0], self.latent_dim), mean=0.0, stddev=1.0)
        z = z_mean + tf.math.exp(z_log_var * .5) * eps
        z_cond = tf.concat([z, input_label], axis=1)  # (batch_size, label_dim + latent_dim)

        return z_cond
