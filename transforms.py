import random
import torch
import numpy as np

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, images, target):
        if random.random() < self.prob:
            images = images.flip(-1)
            height, width = images.shape[-2:]
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return images, target


class ToTensor(object):
    def __call__(self, images, target):
        images = F.to_tensor(images)
        #print(images.shape)
        return images, target


class ConcatImages(object):
    def __call__(self, images, target):

        res = np.array(images[0])
        if res.ndim == 2:
            res = res[:, :, np.newaxis]

        for image in images[1:]:
            image = np.array(image)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            res = np.concatenate((res, image), axis=2)

        # print(res.shape)
        return res, target
