import os
import tensorflow as tf
from model import DCGAN
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("train_size", 15071, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer('input_depth', 12, 'The depth of image to use.')
flags.DEFINE_integer("input_height", 32, "The size of image to use (will be center cropped). [96]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer('output_depth', 12, 'The depth of the output images.')
flags.DEFINE_integer("output_height", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 8, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "taxibj", "The name of dataset [taxibj]")
flags.DEFINE_string('data_type', '', 'Which the data is complete or not')
flags.DEFINE_string('mode', '', 'The mode of convolution')
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def main(_):
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_depth=FLAGS.input_depth,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_depth=FLAGS.output_depth,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=1,
            c_dim=FLAGS.c_dim,
            dataset_name=FLAGS.dataset,
            data_type=FLAGS.data_type,
            mode=FLAGS.mode,
            input_fname_pattern=FLAGS.input_fname_pattern,
            is_crop=FLAGS.is_crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            training=FLAGS.is_train)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
            exit(0)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")

        sample_z = np.random.uniform(-1., 1., size=(10, 1, 100)).astype(np.float32)
        dcgan.same(sample_z, radio=1)
        # visualize(sess, dcgan, FLAGS, option=1)


if __name__ == '__main__':
    tf.app.run()
