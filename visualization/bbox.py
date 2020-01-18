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

# preprocess
dataset = CocoDataset(set_name='val2017')
classes = dataset.labels
val_loader = DataLoader(dataset, batch_size=1, num_workers=12, shuffle=False)

for i, data in tqdm(enumerate(val_loader)):
    ann, img, file_name = data['ann'][0], data['img'][0], data['file_name'][0]
    print(file_name)
    img = cv2.cvtColor(img.numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    for bbox in ann:
        bbox = bbox.int().cpu().numpy()
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        draw_caption(img, (x1, y1, x2, y2), '{}'.format(classes[bbox[4]]))

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
    img = img[:, :, ::-1]
    cv2.imwrite(os.path.join('/home/sensetime/Desktop/result/gt',str(i)+'.jpg'), img)