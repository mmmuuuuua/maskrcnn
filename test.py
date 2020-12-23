import math
import sys
import time
import torch
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

import torchvision.models.detection.mask_rcnn
from dataset import PennFudanDataset
from PIL import Image
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from transform import get_transform
import utils
from model import get_model_instance_segmentation


def get_prediction(pred, threshold):
    # img = Image.open(img_path)
    # transform = T.Compose([T.ToTensor()])
    # img = transform(img)
    # pred = model([img])
    # print('pred')
    # print(pred)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    # print("masks>0.5")
    # print(pred[0]['masks'] > 0.5)
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    print("this is masks")
    # print(masks)
    # pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    # pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(image, output, threshold=0.3, rect_th=3, text_size=1, text_th=3):
    masks, boxes = get_prediction(output, threshold)
    img = cv2.imread("C:\\zhulei\\maskRcnn\\data\\test\\SequenceImages1\\20191227-23-1\\1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, "rock", boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        plt.figure(figsize=(20, 30))
        # plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        # plt.show()
    plt.imsave("C:\\zhulei\\maskRcnn\\results\\img3.png", img)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # load pretrain_dict
    pretrain_dict = torch.load(os.path.join("C:\\zhulei\\maskRcnn\\models", "_epoch-9.pth"))
    model.load_state_dict(pretrain_dict)

    # move model to the right device
    model.to(device)

    # use our dataset and defined transformations
    dataset_test = PennFudanDataset('C:\\zhulei\\maskRcnn\\data\\test', get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader_test.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader_test, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(image)

        instance_segmentation_api(image[0], outputs)

        # 可视化
        # for img in image:
        #     Image.fromarray((img.mul(255).permute(1, 2, 0).byte().cpu().numpy())[0])
        # print(outputs[0]['masks'].shape)
        # for i in range(99):
        #     result = Image.fromarray(outputs[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
        #     result.show()

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)


if __name__ == '__main__':
    main()