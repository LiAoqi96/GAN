import os
import time
import math
from glob import glob
from ops import *
from utils import *
import matplotlib.pyplot as plt


class DCGAN(object):
    def __init__(self, sess, input_depth=8, input_height=96, input_width=96, is_crop=True,
                 batch_size=64, sample_num=64, output_depth=8, output_height=64, output_width=64,
                 z_dim=100, gf_dim=64, df_dim=64, kernel_size=3, gfc_dim=1024, dfc_dim=1024,
                 c_dim=3, dataset_name='default', data_type='complete', mode=None,
                 checkpoint_dir=None, training=True):
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.output_depth = output_depth
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.kernel_size = kernel_size
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir

        self.training = training
        self.d_bn1 = batch_norm(name='d_bn1', train=self.training)
        self.d_bn2 = batch_norm(name='d_bn2', train=self.training)
        if output_height > 32:
            self.d_bn3 = batch_norm(name='d_bn3', train=self.training)

        self.g_bn0 = batch_norm(name='g_bn0', train=self.training)
        self.g_bn1 = batch_norm(name='g_bn1', train=self.training)
        self.g_bn2 = batch_norm(name='g_bn2', train=self.training)
        if output_height > 32:
            self.g_bn3 = batch_norm(name='g_bn3', train=self.training)

        self.build_model()

    def build_model(self):
        if self.mode == '3d':
            image_dim = [self.input_depth, self.input_height, self.input_width, self.c_dim]
        else:
            if self.is_crop:
                image_dim = [self.output_height, self.output_width, self.c_dim]
            else:
                image_dim = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dim, 'real_images')

        inputs = self.inputs
        if self.data_type == 'not':
            inputs = self.filter(inputs)

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], 'z')

        self.G = self.generator(self.z)
        if self.data_type == 'not':
            self.G = self.filter(self.G)
        self.D, self.D_logits = self.discriminator(inputs, reuse=False)
        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        data = self.load_taxibj()

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=0.5) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=0.5) \
            .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run(session=self.sess)

        g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
        d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
        writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1., 1., size=(self.sample_num, self.z_dim)).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(config.epoch):
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in range(batch_idxs):
                batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, d_sum], feed_dict={self.inputs: batch_images, self.z: batch_z})
                writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, g_sum], feed_dict={self.z: batch_z})
                writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([g_optim, g_sum], feed_dict={self.z: batch_z})
                writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f"
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

            if np.mod(epoch, 10) == 9:
                try:
                    samples = self.sess.run(self.sampler, feed_dict={self.z: sample_z})

                    if self.mode == '3d':
                        samples = np.expand_dims(samples[:, 0, :, :, 0], axis=3)
                    else:
                        samples = np.expand_dims(samples[:, :, :, 0], axis=3)
                    save_images(samples, [8, 8],
                                './{}/train_{:02d}.png'.format(config.sample_dir, 4))
                    print('one pic is saved...')
                except Exception as e:
                    print("one pic error!...")
                    print(e)

            if np.mod(epoch, 10) == 9:
                self.save(config.checkpoint_dir, 4)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            if self.output_width > 32:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h4), h4
            elif self.mode == '3d':
                h0 = lrelu(conv3d(image, self.df_dim, k_d=self.kernel_size, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim * 2, k_d=self.kernel_size, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv3d(h1, self.df_dim * 4, k_d=self.kernel_size, name='d_h2_conv')))
                h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h3), h3
            elif self.output_width == 32:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h3), h3

    def filter(self, images):
        with tf.variable_scope('filter'):
            if self.mode == '3d':
                patch = tf.zeros([1, 8, 8, 1], dtype=tf.float32)
                res = []
                for i in range(self.batch_size):
                    d_ = tf.random_uniform([1], minval=0, maxval=11, dtype=tf.int32)[0]
                    rand_num = tf.random_uniform([2], minval=0, maxval=24, dtype=tf.int32)
                    h_, w_ = rand_num[0], rand_num[1]
                    padding = [[d_, 11 - d_], [h_, 32 - h_ - 8], [w_, 32 - w_ - 8], [0, 0]]
                    padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
                    res.append(tf.multiply(images[i], padded) + (1 - padded))
                res = tf.reshape(tf.stack(res), images.get_shape())
                return res
            else:
                patch = tf.zeros([8, 8, 3], dtype=tf.float32)
                res = []
                for i in range(self.batch_size):
                    rand_num = tf.random_uniform([2], minval=0, maxval=24, dtype=tf.int32)
                    h_, w_ = rand_num[0], rand_num[1]
                    c_ = tf.random_uniform([1], minval=0, maxval=9, dtype=tf.int32)[0]

                    padding = [[h_, 32 - h_ - 8], [w_, 32 - w_ - 8], [c_, 9 - c_]]
                    padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)
                    # mask = np.random.choice([1., 0.], [32, 32, 12], p=[0.99, 0.01])
                    # padded = tf.constant(mask, dtype=tf.float32, name='mask')

                    res.append(tf.multiply(images[i], padded) + (1 - padded))
                res = tf.stack(res)
                return res

    def generator(self, z):
        with tf.variable_scope('generator'):
            if self.output_width > 32:
                s_h, s_w = self.output_width, self.output_height
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                z_, h0_w, h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_h16, 'g_h0_lin', with_w=True)
                h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1, h1_w, h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
                                          name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2, h2_w, h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                                          name='g_h2',
                                          with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, h3_w, h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h3',
                                          with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, h4_w, h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4',
                                          with_w=True)
                return tf.nn.tanh(h4)
            elif self.mode == '3d':
                s_d, s_h, s_w = self.output_depth, self.output_width, self.output_height
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

                z_, h0_w, h0_b = linear(z, self.gf_dim * 4 * s_d * s_h8 * s_w8, 'g_h0_lin', with_w=True)
                h0 = tf.reshape(z_, [-1, s_d, s_h8, s_w8, self.gf_dim * 4])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1, h1_w, h1_b = deconv3d(h0, [self.batch_size, s_d, s_h4, s_w4, self.gf_dim * 2],
                                          k_d=self.kernel_size, name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2, h2_w, h2_b = deconv3d(h1, [self.batch_size, s_d, s_h2, s_w2, self.gf_dim], k_d=self.kernel_size,
                                          name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, h3_w, h3_b = deconv3d(h2, [self.batch_size, s_d, s_h, s_w, self.c_dim], k_d=self.kernel_size,
                                          name='g_h3', with_w=True)
                return tf.nn.tanh(h3)
            elif self.output_width == 32:
                s_h, s_w = self.output_width, self.output_height
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

                z_, h0_w, h0_b = linear(z, self.gf_dim * 4 * s_h8 * s_h8, 'g_h0_lin', with_w=True)
                h0 = tf.reshape(z_, [-1, s_h8, s_w8, self.gf_dim * 4])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1, h1_w, h1_b = deconv2d(h0, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                                          name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2, h2_w, h2_b = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim],
                                          name='g_h2',
                                          with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, h3_w, h3_b = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3',
                                          with_w=True)
                return tf.nn.tanh(h3)
            else:
                s_h, s_w = self.input_width, self.input_height
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h2')))
                if self.data_type == 'not':
                    out = tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))
                    patch = tf.zeros([10, 10, 1], dtype=tf.float32)
                    res = []
                    for i in range(self.batch_size):
                        rand_num = tf.random_uniform([2], minval=0, maxval=18, dtype=tf.int32)
                        h_, w_ = rand_num[0], rand_num[1]

                        padding = [[h_, 28 - h_ - 10], [w_, 28 - w_ - 10], [0, 0]]
                        padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

                        res.append(tf.multiply(out[i], padded))
                    res = tf.stack(res)
                    return res
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z):
        batch_size = self.sample_num
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            if self.output_width > 32:
                s_h, s_w = self.output_width, self.output_height
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                z_, h0_w, h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_h16, 'g_h0_lin', with_w=True)
                h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1, h1_w, h1_b = deconv2d(h0, [batch_size, s_h8, s_w8, self.gf_dim * 4],
                                          name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2, h2_w, h2_b = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim * 2],
                                          name='g_h2', with_w=True)

                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, h3_w, h3_b = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim], name='g_h3',
                                          with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, h4_w, h4_b = deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4',
                                          with_w=True)
                return tf.nn.tanh(h4)
            elif self.mode == '3d':
                s_d, s_h, s_w = self.output_depth, self.output_width, self.output_height
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

                z_, h0_w, h0_b = linear(z, self.gf_dim * 4 * s_d * s_h8 * s_w8, 'g_h0_lin', with_w=True)
                h0 = tf.reshape(z_, [-1, s_d, s_h8, s_w8, self.gf_dim * 4])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1, h1_w, h1_b = deconv3d(h0, [batch_size, s_d, s_h4, s_w4, self.gf_dim * 2], k_d=self.kernel_size,
                                          name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2, h2_w, h2_b = deconv3d(h1, [batch_size, s_d, s_h2, s_w2, self.gf_dim], k_d=self.kernel_size,
                                          name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, h3_w, h3_b = deconv3d(h2, [batch_size, s_d, s_h, s_w, self.c_dim], k_d=self.kernel_size,
                                          name='g_h3', with_w=True)
                return tf.nn.tanh(h3)
            elif self.output_width == 32:
                s_h, s_w = self.output_width, self.output_height
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

                z_, h0_w, h0_b = linear(z, self.gf_dim * 4 * s_h8 * s_h8, 'g_h0_lin', with_w=True)
                h0 = tf.reshape(z_, [-1, s_h8, s_w8, self.gf_dim * 4])
                h0 = tf.nn.relu(self.g_bn0(h0))

                h1, h1_w, h1_b = deconv2d(h0, [batch_size, s_h4, s_w4, self.gf_dim * 2],
                                          name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(h1))

                h2, h2_w, h2_b = deconv2d(h1, [batch_size, s_h2, s_w2, self.gf_dim],
                                          name='g_h2',
                                          with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, h3_w, h3_b = deconv2d(h2, [batch_size, s_h, s_w, self.c_dim], name='g_h3',
                                          with_w=True)
                return tf.nn.tanh(h3)

    def load_taxibj(self):
        data_dir = os.path.join("./data", 'taxibj')
        fd = open(os.path.join(data_dir, 'TaxiBJ.bin'))
        loaded = np.fromfile(file=fd, dtype=np.uint16)
        if self.mode == '3d':
            X = loaded.reshape((15071, 12, 32, 32, 1)).astype(np.float32)
        else:
            X = loaded.reshape((15071, -1, 32, 32))
            X = np.transpose(X, (0, 2, 3, 1)).astype(np.float32)
        # np.random.shuffle(X)

        return X / 646. - 1.

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format('model'))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def same(self, z, radio=0.5):
        sample_x = self.load_taxibj()[2]
        mask = generate_mask(sample_x.shape[1], np.sqrt(radio))
        if self.mode == '3d':
            mask = mask.reshape((32, 32, 1))
            results = np.append(sample_x[0], sample_x[0] * mask, axis=0)
        else:
            results = np.append(sample_x[:, :, 0], sample_x[:, :, 0] * mask, axis=0)
        if radio == 1:
            a = 1
            b = 1
        else:
            a = int(radio * 8 + 0.5)
            b = 0
        sample = self.sampler
        if self.mode == '3d':
            loss = tf.sqrt(0.2 * tf.reduce_sum((sample[0, 1:] - sample_x[1:]) ** 2) +
                           tf.reduce_sum((sample[0, 0] - sample_x[0]) ** 2 * mask))
        else:
            loss = tf.sqrt(0.2 * tf.reduce_sum((sample[0, :, :, a:] - sample_x[:, :, a:]) ** 2) +
                           b * tf.reduce_sum((sample[0, :, :, 0] - sample_x[:, :, 0]) ** 2 * (1 - mask)))

        gradient = tf.gradients(loss, self.z)[0]
        loss_sum = []

        result_z = np.zeros((1, 100))
        result_s = np.zeros((1, 100))
        min_loss = 10
        for i in range(z.shape[0]):
            s = np.zeros((1, 100), dtype=np.float32)
            for j in range(100):
                grd = self.sess.run(gradient, feed_dict={self.z: z[i]})
                z[i], s = update_z(grd, z[i], s)
            diff = self.sess.run(loss, feed_dict={self.z: z[i]})
            if diff < min_loss:
                min_loss = diff
                result_z = z[i]
                result_s = s

        for i in range(1000):
            grd, diff = self.sess.run([gradient, loss], feed_dict={self.z: result_z})
            loss_sum.append(diff)
            result_z, result_s = update_z(grd, result_z, result_s)
            if i % 100 == 0:
                print(str(i) + '  ', diff)
            if i % 100 == 99:
                y = self.sess.run(sample, feed_dict={self.z: result_z})
                if self.mode == '3d':
                    result = (y[0, 0] * (1 - mask) + sample_x[0] * mask).reshape((32, 32, 1))
                else:
                    result = y[0, :, :, 0] * (1 - mask) + sample_x[:, :, 0] * mask
                results = np.append(results, result, axis=0)

        plt.plot(loss_sum)
        plt.show()
        results = results.reshape((-1, 32, 32, 1))
        save_images(results, [3, 4], './{}/result_{:02d}.png'.format('./samples', 2000))

        if self.mode == '3d':
            result = self.sess.run(sample, feed_dict={self.z: result_z})[0, 0, :, :, 0]
            sample_x = sample_x[0, :, :, 0]
        else:
            result = self.sess.run(sample, feed_dict={self.z: result_z})[0, :, :, 0]
            sample_x = sample_x[:, :, 0]
        RMSE = compute_rmse(result, sample_x, mask)
        print(RMSE)


def conv_out_size_same(size, stride):
    return math.ceil(float(size) / float(stride))


def update_z(grd, z, s, lr=0.002, beta2=0.9, epsilon=1e-8):
    s = beta2 * s + (1 - beta2) * grd ** 2
    return z - lr * grd / np.sqrt(s + epsilon), s


def generate_mask(length, radio=0.5):
    if radio < 1:
        a = int(length * radio)
        p1, p2 = np.random.randint(0, length - a), np.random.randint(0, length - a)
        p3, p4 = length - a - p1, length - a - p2
        mask = np.zeros([a, a])
        mask = np.pad(mask, ((p1, p3), (p2, p4)), 'constant', constant_values=1)
    else:
        mask = np.zeros((length, length))
    return mask


def compute_rmse(result, x, mask):
    RMSE = np.sqrt(np.mean((np.uint16((result + 1.) * 646.) - np.uint16((x + 1.) * 646.)) ** 2))
    return RMSE
