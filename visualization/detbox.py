"""
visualization the result of object detection.
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ..dataloader.coco import CocoDataset

def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
def show2tensorboard(scores, classification, transformed_anchors, img, labels, writer):
    """
    show the result of object detection in tensorboard.
    """
    idxs = np.where(scores > 0.5)
    img = np.array(255 * img).copy()

    img[img < 0] = 0
    img[img > 255] = 255

    img = np.transpose(img, (1, 2, 0))

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        label_name = labels[int(classification[idxs[0][j]])]
        score = scores[idxs[0][j]]
        draw_caption(img, (x1, y1, x2, y2), '{}:{:.2f}'.format(label_name, score.item()))

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        writer.add_image(label_name, img.transpose(2, 0, 1)[::-1])

    return

def show2img(scores, classification, transformed_anchors, img, labels):
    """
    show the result of object detection.
    """
    idxs = np.where(scores > 0.5)
    img = np.array(255 * img).copy()

    img[img < 0] = 0
    img[img > 255] = 255

    img = np.transpose(img, (1, 2, 0))

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        label_name = labels[int(classification[idxs[0][j]])]
        score = scores[idxs[0][j]]
        draw_caption(img, (x1, y1, x2, y2), '{}:{:.2f}'.format(label_name, score.item()))

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imshow(label_name, img.transpose(2, 0, 1)[::-1])
    cv2.waitKey(0)

def show2img(scores, classification, transformed_anchors, img, labels):
    """
    show the result of object detection.
    """
    idxs = np.where(scores > 0.5)
    img = np.array(255 * img).copy()

    img[img < 0] = 0
    img[img > 255] = 255

    img = np.transpose(img, (1, 2, 0))

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        label_name = labels[int(classification[idxs[0][j]])]
        score = scores[idxs[0][j]]
        draw_caption(img, (x1, y1, x2, y2), '{}:{:.2f}'.format(label_name, score.item()))

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imwrite(os.path.join('./','detbox'+'.jpg'), img.transpose(2, 0, 1)[::-1])
    cv2.waitKey(0)
