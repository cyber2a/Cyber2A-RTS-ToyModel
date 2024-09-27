import json
import os

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms.v2 import functional as F


class RTSDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms=None):
        self.root = "data/images"
        with open(data_path) as f:
            coco_data = json.load(f)
        self.images = coco_data["images"]
        self.annotations = self._load_annotations(coco_data["annotations"])
        self.transforms = transforms

    def _load_annotations(self, annotations):
        anno_dict = {}
        for anno in annotations:
            if anno["image_id"] not in anno_dict:
                anno_dict[anno["image_id"]] = []
            anno_dict[anno["image_id"]].append(anno)
        return anno_dict

    def __getitem__(self, index):
        image_info = self.images[index]
        image = read_image(os.path.join(self.root, image_info["file_name"]))

        annotation = self.annotations[image_info["id"]]
        masks, num_objs = self._create_masks(annotation, F.get_size(image))

        boxes = masks_to_boxes(masks)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image = tv_tensors.Image(image)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(image)
            ),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": index,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _create_masks(self, annotation, image_size):
        masks = np.zeros(image_size, dtype=np.uint8)
        for i, anno in enumerate(annotation):
            contour = np.array(anno["segmentation"]).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(masks, [contour], i + 1)
        masks = torch.from_numpy(masks)
        obj_ids = torch.unique(masks)[1:]
        num_objs = len(obj_ids)
        masks = (masks == obj_ids[:, None, None]).to(dtype=torch.uint8)
        return masks, num_objs

    def __len__(self):
        return len(self.images)
