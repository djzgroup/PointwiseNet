import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from datetime import datetime
from functools import reduce
from operator import mul
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import provider

Data_Path = '/home/hi/Dataset/Pointdata/'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024/2048/5000] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum or Adagrad or RMS [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--normal', type=bool, default=False, help='Whether to use normal information')

parser.add_argument('--KNN', type=int, default=16, help='KNN [3/6/9/12/16/24] [default: 16]')
parser.add_argument('--node_num', type=int, default=256, help='node number [128/256/384/512/1024] [default: 256]')
parser.add_argument('--topk', type=int, default=3, help='topk [1/2/3/4] [default: 3]')
parser.add_argument('--centers', type=int, default=32, help='centers number [default: 32]')
parser.add_argument('--STN', type=bool, default=True, help='Whether to use STN')
parser.add_argument('--VLAD', type=bool, default=True, help='Whether to use VLAD')

parser.add_argument('--local_pool', type=str, default='maxpool', help='Local pool: average or maxpool [default: maxpool]')
parser.add_argument('--global_pool', type=str, default='maxpool', help='Global pool: average or maxpool [default: maxpool]')

FLAGS = parser.parse_args()

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp train.py %s' % (LOG_DIR))
os.system('cp models/VLAD_layer.py %s' % (LOG_DIR))
os.system('cp models/transform_nets.py %s' % (LOG_DIR))
os.system('cp models/pointnet_cls.py %s' % (LOG_DIR))

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(Data_Path, 'data/ModelNet40_normal/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(Data_Path, 'data/ModelNet40_normal/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_params_num():
    num_params = 0
    for varable in tf.trainable_variables():
        shape = varable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def train():
    with tf.Graph().as_default() as graph:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, index = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay,num_class=NUM_CLASSES, FLAGS=FLAGS)
            loss = MODEL.get_loss(pred, labels_pl, NUM_CLASSES)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif OPTIMIZER == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        stats_graph(graph)

        num_params = get_params_num()
        print('num_params:', num_params)
        log_string('num_params: %s' % num_params)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'index': index}

        acc_record = 0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch+1))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            acc = eval_one_epoch(sess, ops, test_writer)

            if acc > acc_record:
                acc_record = acc
                saver.save(sess, os.path.join('log/best_model', 'model.ckpt'))

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
                log_string("Model saved in file: %s" % save_path)
            print("current best acc: %s" % acc_record)
        print("best acc: %s" % acc_record)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    print(str(datetime.now()))

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))

    current_data, current_label, normal_data = provider.loadDataFile_with_normal(
        Data_Path + TRAIN_FILES[train_file_idxs[0]])
    normal_data = normal_data[:, 0:NUM_POINT, :]
    current_data = current_data[:, 0:NUM_POINT, :]
    current_data, current_label, shuffle_idx = provider.shuffle_data(current_data, np.squeeze(current_label))
    current_label = np.squeeze(current_label)
    normal_data = normal_data[shuffle_idx, ...]

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        jittered_data = provider.jitter_point_cloud(rotated_data)
        input_data = np.concatenate((jittered_data, normal_data[start_idx:end_idx, :, :]), 2)
        # random point dropout
        input_data = provider.random_point_dropout(input_data)

        feed_dict = {ops['pointclouds_pl']: input_data,
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                         feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

        if (batch_idx + 1) % 60 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss: %f' % (loss_sum / 60))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    current_data, current_label, normal_data = provider.loadDataFile_with_normal(Data_Path + TEST_FILES[0])
    normal_data = normal_data[:, 0:NUM_POINT, :]
    current_data = current_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(current_label)
    input_data = np.concatenate((current_data, normal_data), 2)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT+1))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: input_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val * BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    EPOCH_CNT += 1

    return (total_correct / float(total_seen))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
