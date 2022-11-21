#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import json
import os
import tempfile

from yolox.utils import xyxy2xywh
from pycocotools.coco import COCO

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class JointEvaluator:

    def __init__(
        self,
        annotation_file=None,
        img_size: int = 640,
        confthre: float = 0.3,
        nmsthre: float = 0.3,
        num_classes: int = 80,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
    ):
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        with suppress_stdout_stderr():
            self.coco = COCO(annotation_file) if annotation_file != None else COCO()
            # self.coco = COCO()

    def convert_to_coco_format(self, outputs, h0, w0, ids):
        data_list = []
        # *_, img_h, img_w = info_imgs.shape
        img_h = h0
        img_w = w0
        for (output, img_id) in zip(
                outputs, ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size / float(img_h), self.img_size / float(img_w)
            )
            pad = (self.img_size - img_w * scale) / 2, (self.img_size - img_h * scale) / 2  # wh padding
            bboxes[:, [0, 2]] -= pad[0]
            bboxes[:, [1, 3]] -= pad[1]
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": int(cls[ind]) + 1,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        return data_list

    def evaluate_prediction(self, data_dict, gt_data_dict):
        annType = ["segm", "bbox", "keypoints"]

        if len(data_dict) > 0:
            cocoGt = self.coco
            _, tmp = tempfile.mkstemp()
            # _, tmp_gt = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            # json.dump(gt_data_dict, open(tmp_gt, "w"))

            cocoDt = cocoGt.loadRes(tmp)

            from pycocotools.cocoeval import COCOeval

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            # redirect_string = io.StringIO()
            # with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()

# if __name__ == '__main__':
#     json_file = os.path.join('/home/xteam/hj/dataset/LOLDataset', 'eval15/eval.json')
#     evaluator = JointEvaluator(annotation_file=json_file)