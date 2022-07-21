#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from .coco import COCODataset


def scale_annotations(annotations, original_scale, new_scale):
    scaled_annotations = []
    for annot in annotations:
        x_min, y_min, x_max, y_max, cls = annot
        x, y, w, h = x_min, y_min, (x_max - x_min), (y_max - y_min)
        h, w, x, y = _process_scale(h, new_scale, original_scale, w, x, y)
        scaled_annotations.append(torch.tensor([x, y, x + w, y + h, cls]))
    out = torch.stack(scaled_annotations) if len(scaled_annotations) != 0 else []
    return out


def _process_scale(h, new_scale, original_scale, w, x, y):
    # o_w, o_h = original_scale
    # c_w, c_h = new_scale
    o_h, o_w = original_scale
    c_h, c_w = new_scale
    x = (c_w * x) // o_w  # torch.div((c_w * x), o_w, rounding_mode='floor')  # (c_w * x) // o_w
    y = (c_h * y) // o_h
    w = (c_w * w) // o_w
    h = (c_h * h) // o_h
    return h, w, x, y


def verify_loaded_annotations(imgs_root, annotations):
    for annotation in annotations:
        bbox, img_info, resized_info, img_file_name = annotation
        img = Image.open(os.path.join(imgs_root, img_file_name))
        img_o = np.asarray(img)
        img = cv2.resize(img_o, dsize=(resized_info[1], resized_info[0]), interpolation=cv2.INTER_CUBIC)
        if len(bbox) == 0:
            continue
        for box in bbox:
            x1, y1, x2, y2, clz = box.astype(int)
            p1, p2 = (x1, y1), (x2, y2)
            cv2.rectangle(img, p1, p2, (255, 100, 25), 2, lineType=cv2.LINE_AA)
        plt.imshow(img)
        plt.show()
        print()


class OOCCOCODataset(COCODataset):
    """
    COCO dataset class.
    """

    def __init__(
            self,
            data_dir=None,
            json_file="train_coco.json",
            name="train",
            img_size=(416, 416),
            preproc=None,
            cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(
            data_dir=data_dir,
            json_file=json_file,
            name=name,
            img_size=img_size,
            preproc=preproc,
            cache=cache,
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids, annotations = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        # annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if "area" not in obj:
                obj["area"] = obj["bbox"][2] * obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        # resized_annotations = scale_annotations(res, img_info, resized_info)
        # resized_annotations = resized_annotations.numpy() if len(resized_annotations) != 0 else resized_annotations

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return res, img_info, resized_info, file_name
