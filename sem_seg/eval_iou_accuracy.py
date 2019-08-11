import numpy as np
import os

pred_data_label_filenames = [line.rstrip() for line in open('all_pred_data_label_filelist.txt')]

gt_label_filenames = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_filenames]
num_room = len(gt_label_filenames)

LOG_FOUT = open('eval_iou.txt', 'w')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]
for i in range(num_room):
    log_string(str(i))
    data_label = np.loadtxt(pred_data_label_filenames[i])
    pred_label = data_label[:,-1]
    gt_label = np.loadtxt(gt_label_filenames[i])
    log_string(str(gt_label.shape))
    for j in range(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)


log_string(str(gt_classes))
log_string(str(positive_classes))
log_string(str(true_positive_classes))


log_string('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

log_string('IoU:')
iou_list = []
for i in range(13):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    log_string(str(iou))
    iou_list.append(iou)

log_string("mean IoU: %s" % str(sum(iou_list)/13.0))