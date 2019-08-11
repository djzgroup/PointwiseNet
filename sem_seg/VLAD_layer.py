import tensorflow as tf
from utils.tf_util import _variable_on_cpu, conv2d
import numpy as np


def Top_K(one_hot, top_k, batch_size, num_point):
    first = tf.constant(np.arange(batch_size).reshape(batch_size, 1))
    first = tf.tile(first, [1, num_point])
    first = tf.expand_dims(first, axis=-1)
    second = tf.constant(np.arange(num_point).reshape(num_point, 1))
    second = tf.tile(second, [batch_size, 1])
    second = tf.reshape(second, shape=[batch_size, -1])
    second = tf.expand_dims(second, axis=-1)

    concat = tf.concat([first, second], axis=2)
    concat = tf.tile(concat, [1, 1, top_k])
    concat = tf.reshape(concat, shape=[batch_size, num_point, top_k, -1])
    concat = tf.cast(concat, dtype=tf.int32)
    one_hot = tf.cast(one_hot[1], dtype=tf.int32)
    one_hot = tf.reshape(one_hot, shape=[batch_size, num_point, top_k, -1])
    fuse = tf.concat([concat, one_hot], axis=3)

    return fuse


def VLAD(input_tensor, K, is_training, bn_decay, layer_name=None):

    with tf.variable_scope('VLAD_layer'):
        batch_size = input_tensor.get_shape()[0].value
        num_point = input_tensor.get_shape()[1].value
        D = input_tensor.get_shape()[-1].value
        top_k = 1

        reshape = tf.reshape(input_tensor, shape=[batch_size, num_point, D])
        conv_norm = tf.nn.l2_normalize(reshape, dim=2)
        descriptor = tf.expand_dims(conv_norm, axis=-1, name='expanddim')  # descriptor is B x N x D x 1
        conv = conv2d(descriptor, K, [1, D],
                      padding='VALID', stride=[1, 1],
                      bn=True, is_training=is_training,
                      scope='assignment', bn_decay=bn_decay,
                      activation_fn=None)
        a = tf.nn.softmax(conv)
        one_hot = tf.nn.top_k(a, top_k)
        fuse = Top_K(one_hot, top_k, batch_size, num_point)

        Center = _variable_on_cpu(name=layer_name+'_centers',
                                  shape=[D, K],
                                  initializer=tf.contrib.layers.xavier_initializer())

        tf.add_to_collection(tf.GraphKeys.WEIGHTS, Center)
        tf.summary.histogram('VLAD_center', Center)

        diff = tf.expand_dims(reshape, axis=-1) - Center
        diff = diff * a
        diff = tf.transpose(diff, perm=[0,1,3,2])
        diff = tf.gather_nd(diff, fuse)

        net = conv2d(diff, 128, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv1', bn_decay=bn_decay)
        net = conv2d(net, 256, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv2', bn_decay=bn_decay)

        concat = tf.concat([input_tensor, net], axis=-1)

        net = conv2d(concat, 384, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv3', bn_decay=bn_decay)
        net = conv2d(net, 1024, [1, 1],
                     padding='VALID', stride=[1, 1],
                     bn=True, is_training=is_training,
                     scope='vlad_conv4', bn_decay=bn_decay)

        net = tf.nn.max_pool(net, ksize=[1, num_point, 1, 1], strides=[1, 2, 2, 1], padding='VALID')
        net = tf.reshape(net, [batch_size, -1])

        return net, one_hot[1]
