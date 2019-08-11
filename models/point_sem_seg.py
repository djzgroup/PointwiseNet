import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/grouping'))
from utils import tf_util
from models.transform_nets import input_transform_net
from models.VLAD_layer import VLAD_part
from models.point_util import sample_and_group, pointnet_fp_module


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(inputs, is_training, bn_decay=None):
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value
    point_cloud = inputs[:, :, 0:3]
    points = inputs[:, :, 3:]

    with tf.variable_scope('transform_net') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    with tf.variable_scope('group_sampling') as sc:
        KNN = 16
        Node_NUM = 384
        point_cloud_sampled, nn_points, _, _ = sample_and_group(npoint=Node_NUM, radius=0.2, nsample=KNN,
                                                                xyz=point_cloud_transformed, points=points, knn=True,
                                                                use_xyz=True)

    net1 = tf_util.conv2d(tf.expand_dims(point_cloud_sampled, axis=-1), 64, [1, 3],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='conv1_1', bn_decay=bn_decay)
    net1 = tf.tile(net1, multiples=[1, 1, KNN, 1])
    net1 = tf.expand_dims(net1, axis=-2)

    nn_points = tf.expand_dims(nn_points, axis=-1)
    net = tf_util.conv3d(nn_points, 64, [1, 1, 9],
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
    pool_k = tf_util.max_pool3d(net, kernel_size=[1, KNN, 1],
                                stride=[1, 2, 2], padding='VALID', scope='pool_k')
    net = tf.squeeze(pool_k, axis=2)

    # vlad layer
    vlad_out, index = VLAD_part(net, 13, is_training, bn_decay, layer_name='VLAD')

    vlad_out = tf_util.conv2d(vlad_out, 384, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='vlad_conv3', bn_decay=bn_decay)
    vlad_out = tf_util.conv2d(vlad_out, 512, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=is_training,
                              scope='vlad_conv4', bn_decay=bn_decay)
    out_max = tf.nn.max_pool(vlad_out, ksize=[1, Node_NUM, 1, 1], strides=[1, 2, 2, 1],
                             padding='VALID')

    expand = tf.tile(out_max, multiples=[1, Node_NUM, 1, 1])
    concat = tf.concat([expand, vlad_out, net], axis=-1)

    concat = tf_util.conv2d(concat, 512, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv4', bn_decay=bn_decay)
    concat = tf_util.conv2d(concat, 256, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)
    concat = tf.squeeze(concat, axis=2)

    # segmentation network
    l0_points = pointnet_fp_module(xyz1=point_cloud_transformed, xyz2=point_cloud_sampled, points1=tf.concat([point_cloud_transformed, points], axis=-1),
                                   points2=concat, mlp=[128, 128], is_training=is_training, bn_decay=bn_decay, scope='layer6')

    l0_points = tf.expand_dims(l0_points, axis=2)
    net = tf_util.conv2d(l0_points, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                         scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 13, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='fc2')
    net = tf.squeeze(net, axis=2)

    return net


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)