import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import json
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import tf_util
import part_dataset_all_normal


Data_Path = '/home/hi/Dataset/Pointdata/'
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='point_part_seg', help='Model name [default: point_part_seg]')
parser.add_argument('--model_path', default='log85.44/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='log_visual', help='Log dir [default: log_visual]')
FLAGS = parser.parse_args()

output_verbose = True   # If true, output all color-coded part segmentation obj files

VOTE_NUM = 12

EPOCH_CNT = 0

BATCH_SIZE = 1
NUM_POINT = 2048
GPU_INDEX = FLAGS.gpu

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
NUM_CLASSES = 50
NUM_OBJ_CATS = 16

hdf5_data_dir = './meta'
oid2cpid = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

object2setofoid = {}
for idx in range(len(oid2cpid)):
    objid, pid = oid2cpid[idx]
    if not objid in object2setofoid.keys():
        object2setofoid[objid] = []
    object2setofoid[objid].append(idx)

all_obj_cat_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cat_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
objcats = [line.split()[1] for line in lines]
objnames = [line.split()[0] for line in lines]
on2oid = {objcats[i]:i for i in range(len(objcats))}
fin.close()

color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))

cpid2oid = json.load(open(os.path.join(hdf5_data_dir, 'catid_partid_to_overallid.json'), 'r'))

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            idx = seg[i]
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        seg = seg[0]
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

# Shapenet official train/test split
DATA_PATH = os.path.join(Data_Path, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test')

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, cls_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            print("--- Get model and loss")
            pred, _ = MODEL.get_model(pointclouds_pl, cls_labels_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'cls_labels_pl': cls_labels_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}

        eval_one_epoch(sess, ops)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_cls_label = np.zeros((bsize,), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg,cls = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
        batch_cls_label[i] = cls
    return batch_data, batch_label, batch_cls_label


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET) + BATCH_SIZE - 1) / BATCH_SIZE

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx + 1) * BATCH_SIZE)
        cur_batch_size = end_idx - start_idx
        cur_batch_data, cur_batch_label, cur_batch_cls_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        batch_data[0:cur_batch_size] = cur_batch_data
        batch_label[0:cur_batch_size] = cur_batch_label

        # ---------------------------------------------------------------------
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['cls_labels_pl']: cur_batch_cls_label,
                     ops['is_training_pl']: is_training}
        loss, seg_pred_res = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        # ---------------------------------------------------------------------

        seg_pred_res = seg_pred_res[0, ...]

        iou_oids = object2setofoid[objcats[cur_batch_cls_label[0]]]
        non_cat_labels = list(set(np.arange(NUM_CLASSES)).difference(set(iou_oids)))

        mini = np.min(seg_pred_res)
        seg_pred_res[:, non_cat_labels] = mini - 1000

        seg_pred_val = np.argmax(seg_pred_res, axis=1)[:cur_batch_data.shape[1]]

        if output_verbose:
            file = str(cur_batch_cls_label[0])
            if not os.path.exists(LOG_DIR+'/'+file): os.mkdir(LOG_DIR+'/'+file)

            output_color_point_cloud(cur_batch_data[0, :, 0:3], batch_label[0], os.path.join(LOG_DIR, file + '/' + str(batch_idx) + '_gt.obj'))
            output_color_point_cloud(cur_batch_data[0, :, 0:3], seg_pred_val, os.path.join(LOG_DIR, file + '/' + str(batch_idx) + '_pred.obj'))
            output_color_point_cloud_red_blue(cur_batch_data[0, :, 0:3], np.int32(batch_label == seg_pred_val),
                                              os.path.join(LOG_DIR, file + '/' + str(batch_idx) + '_diff.obj'))




if __name__=='__main__':
    evaluate()
