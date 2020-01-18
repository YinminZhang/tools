"""
load segmentation mask.
"""
import numpy as np
import os
import json
import torch
import torch.nn.functional as F

def load_weights(img_meta, set_name='weight_linear_train', dir_root='./data/weight'):
    """
    load segmentation mask, and convert to tensor in gpu.
    """
    weight_list = []
    pad_shape_list = []
    for i in img_meta:
        pad_shape = i['pad_shape']
        pad_shape_list.append(pad_shape)
    pad_shape = np.array(pad_shape_list).max(axis=0)
    for i in img_meta:
        file_name = i['filename'].split('/')[-1].replace('jpg', 'json')
        img_shape = i['img_shape']
        pad = (0, pad_shape[1] - img_shape[1],
               0, pad_shape[0] - img_shape[0])
        with open(os.path.join(dir_root, set_name, file_name), 'r') as f:
            load_dict = json.load(f)
            weight = json.loads(load_dict)['file_name']
            if len(weight):
                weight = F.interpolate(torch.tensor(weight)[None, None, :, :].float(), size=img_shape[:-1], mode='nearest')
                weight = F.pad(weight, pad=pad, mode='replicate')[0, :, :, :].tolist()
                weight_list.append(weight)
            else:
                weight_list.append(np.ones(pad_shape[:-1])[ None, :, :].tolist())

    weight = torch.tensor(weight_list).cuda()
    return weight

def gaussian_mask(bbox, min_overlap):
    ctr_x = (bbox[2] + bbox[0]) / 2
    ctr_y = (bbox[3] + bbox[1]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    bbox = bbox.int()
    return bbox[0], bbox[2], bbox[1], bbox[3]


def bbox2mask(gt_bboxes, img_meta):
    """
    implement bboxes converted to mask.
    """
    batch_size = len(gt_bboxes)
    min_overlap = 0.5
    pad_shape = []

    # get the image shape of input (N, C, Hmax, Wmax)
    for i in img_meta:
        pad_shape.append(i['pad_shape'])
    pad_shape = torch.tensor(pad_shape).max(dim=0)[0][:2].tolist()

    mask_list = []
    for i, bboxes in enumerate(gt_bboxes):
        # img_shape = img_meta[i]['img_shape']
        mask = torch.zeros(size=pad_shape)[None, :, :]
        for bbox in bboxes:
            x1, x2, y1, y2 = random_shift(bbox, 0.3).astype(np.int)
            mask[:, y1:y2, x1:x2] = 1
        mask_list.append(mask)

    mask = torch.stack(mask_list).cuda()
    return mask

def mask2onehot(mask):
    N, C, H, W = mask.shape
    one_hot = torch.where(mask==0, torch.zeros(size=(N, C*2, H, W)).cuda(), torch.ones(size=(N, C*2, H, W)).cuda())
    return one_hot.view(N, -1, H, W).float()

def transform_scale(w, h, scale=1.2):
    """
    use scale factory to change the weight and height of object.
    """
    if h < 32 and w < 32:
        return  w * scale, h * scale
    else:
        return w, h

def random_shift(bbox, ratio=0.1, gaussian=False):
    """
    add noise to bbox, to bridge the gap of training and inference.
    """
    ctr_x = (bbox[2] + bbox[0]) / 2
    ctr_y = (bbox[3] + bbox[1]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    w, h = transform_scale(w, h)
    if gaussian:
        ctr_x_shift = ctr_x + truncated_normal() * w * ratio
        ctr_y_shift = ctr_y + truncated_normal() * h * ratio
        w = w * (1 + truncated_normal() * ratio)
        h = h * (1 + truncated_normal() * ratio)
    else:
        ctr_x_shift = ctr_x + (np.random.random() * 2 - 1) * w * ratio
        ctr_y_shift = ctr_y + (np.random.random() * 2 - 1) * h * ratio
        w = w * (1 + (np.random.random() * 2 - 1) * ratio)
        h = h * (1 + (np.random.random() * 2 - 1) * ratio)
    return np.array([ctr_x_shift - w / 2.0, ctr_y_shift - h / 2.0,
                     ctr_x_shift + w / 2.0, ctr_y_shift + h / 2.0])

def truncated_normal(mean=0, std=1, minval=-1, maxval=1):
    """
    generate truncat gaussian distribution.
    """
    return np.clip(np.random.normal(mean, std), minval, maxval)