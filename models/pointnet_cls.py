import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from utils import tf_util
from models.transform_nets import input_transform_net
from models.VLAD_layer import VLAD
from models.point_util import sample_and_group

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(inputs, is_training, bn_decay=None, num_class=40, FLAGS=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    point_cloud = inputs[:, :, 0:3]
    if FLAGS.normal:
        D = 6
        points = inputs[:, :, 3:]
    else:
        D = 3
        points = None

    # --------------------------------------- STN -------------------------------------
    if FLAGS.STN:
        with tf.variable_scope('transform_net') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud = tf.matmul(point_cloud, transform)

    # ---------------------------------- Node Sampling --------------------------------
    with tf.variable_scope('group_sampling') as sc:
        KNN = FLAGS.KNN
        point_cloud_sampled, nn_points, _, _ = sample_and_group(npoint=FLAGS.node_num, radius=0.2, nsample=KNN,
                                                                xyz=point_cloud, points=points, knn=True, use_xyz=True)

    point_cloud_sampled = tf.expand_dims(point_cloud_sampled, axis=-1)
    net1 = tf_util.conv2d(point_cloud_sampled, 64, [1, 3],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='conv1_1', bn_decay=bn_decay)
    net1 = tf.tile(net1, multiples=[1, 1, KNN, 1])
    net1 = tf.expand_dims(net1, axis=-2)

    nn_points = tf.expand_dims(nn_points, axis=-1)
    net = tf_util.conv3d(nn_points, 64, [1, 1, D],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1_2', bn_decay=bn_decay)
    concat = tf.concat(values=[net, net1], axis=-1)

    net = tf_util.conv3d(concat, 128, [1, 1, 1],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 128, [1, 1, 1],
                         padding='VALID', stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)

    # ---------------------- local pooling: merge local feature -------------------------
    if FLAGS.local_pool == 'average':
        pool_k = tf_util.avg_pool3d(net, kernel_size=[1, KNN, 1],
                                    stride=[1, 2, 2], padding='VALID', scope='pool_k')
    else:
        pool_k = tf_util.max_pool3d(net, kernel_size=[1, KNN, 1],
                                    stride=[1, 2, 2], padding='VALID', scope='pool_k')
    net = tf.squeeze(pool_k, axis=2)

    # ---------------------------------- VLAD layer --------------------------------------
    net, index = VLAD(net, FLAGS, is_training, bn_decay, layer_name='VLAD')
    
    # -------------------------------- classification ------------------------------------
    with tf.name_scope('fc_layer'):
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp2')
        net = tf_util.fully_connected(net, num_class, activation_fn=None, scope='fc3')

    return net, index


def get_loss(pred, label, num_class):
    labels = tf.one_hot(indices=label, depth=num_class)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)

    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
