import json

from yolox.data import COCO_CLASSES
from yolox.utils import xyxy2xywh


class COCO_JSON:
    def __init__(self, tsize=640):
        self.img_size = tsize
        self.categories = []
        self.write_json_context = dict()
        self.get_categories()
        self.get_json_context()
        self.image_idx = 0

    def add_coco_format(self, outputs, h0, w0, ids):
        img_h = h0
        img_w = w0
        for (output, img_id) in zip(
                outputs, ids
        ):
            img_context = {}
            img_context['file_name'] = img_id + '.png'
            img_context['height'] = h0
            img_context['width'] = w0
            img_context['date_captured'] = '2022-07-8'
            img_context['id'] = int(img_id)  # 该图片的id
            img_context['license'] = 1
            img_context['color_url'] = ''
            img_context['flickr_url'] = ''
            self.write_json_context['images'].append(img_context)

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
                bbox_dict = {}
                bbox_dict['id'] = int(self.image_idx * 10000 + ind)
                bbox_dict['image_id'] = int(img_id)
                bbox_dict['category_id'] = int(cls[ind]) + 1
                bbox_dict['iscrowd'] = 0
                bbox_dict['area'] = h0 * w0
                bbox_dict['bbox'] = bboxes[ind].numpy().tolist()
                bbox_dict['score'] = scores[ind].numpy().item()
                bbox_dict['segmentation'] = []
                self.write_json_context['annotations'].append(bbox_dict)

            self.image_idx += 1

    def save_json_file(self, save_path):
        with open(save_path, 'w') as fw:
            json.dump(self.write_json_context, fw, indent=2)

    def get_categories(self):
        for j, label in enumerate(COCO_CLASSES):
            self.categories.append({'id': j + 1, 'name': label, 'supercategory': 'None'})

    def get_json_context(self):

        self.write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2022,
                                           'contributor': 'hj', 'date_created': '2022-11-1'}
        self.write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
        self.write_json_context['categories'] = self.categories
        self.write_json_context['images'] = []
        self.write_json_context['annotations'] = []
