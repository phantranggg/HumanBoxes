###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: March 7th 2019                                                 #
###########################################################################################

import os
from metrics.lib.BoundingBox import BoundingBox
from metrics.lib.BoundingBoxes import BoundingBoxes
from metrics.lib.Evaluator import *
from metrics.lib.utils import BBFormat, CoordinatesType
from metrics.lib import my_utils


def get_bounding_box(id2boxes):
    bounding_boxes = BoundingBoxes()
    for id, boxes in id2boxes.items():
        for box in boxes:
            if box["type"] == "gt":
                bb_type = BBType.GroundTruth
                confidence = None
            else:
                bb_type = BBType.Detected
                confidence = box["confidence"]

            bb = BoundingBox(
                imageName=id,
                classId=box["class_id"],
                x=box["left"],
                y=box["top"],
                w=box["width"],
                h=box["height"],
                bbType=bb_type,
                classConfidence=confidence,
                format=BBFormat.XYWH
            )

            bounding_boxes.addBoundingBox(bb)

    return bounding_boxes


def evaluate(id2gt, id2det, iou_threshold=0.5):
    gt_bounding_boxes = get_bounding_box(id2gt)
    det_bounding_boxes = get_bounding_box(id2det)
    bounding_box_list = gt_bounding_boxes.getBoundingBoxes()
    bounding_box_list.extend(det_bounding_boxes.getBoundingBoxes())

    all_bounding_boxes = BoundingBoxes()
    for bb in bounding_box_list:
        all_bounding_boxes.addBoundingBox(bb)

    evaluator = Evaluator()
    acc_AP = 0
    num_valid_classes = 0

    # Plot Precision x Recall curve
    results = evaluator.GetPascalVOCMetrics(
        boundingboxes=all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iou_threshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)

    # each element corresponding with each class
    class2ap = {}
    print("\n------- Evaluate Result -------\n")
    for result in results:

        # Get metric values per each class
        cl = result['class']
        ap = result['AP']
        precision = result['precision']
        recall = result['recall']
        total_positive = result['total positives']
        total_TP = result['total TP']
        total_FP = result['total FP']

        if total_positive > 0:
            num_valid_classes = num_valid_classes + 1
            acc_AP = acc_AP + ap
            # prec = ['%.2f' % p for p in precision]
            # rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))

            class2ap[cl] = ap

    mAP = acc_AP / num_valid_classes
    class2ap["mAP"] = mAP
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)

    return class2ap


if __name__ == "__main__":
    gt_dir = os.path.abspath("./groundtruths/")
    det_dir = os.path.abspath("./detections/")

    id2gt, id2det = {}, {}
    for dir, type in [(gt_dir, "gt"), (det_dir, "det")]:
        for img_name in my_utils.get_file_names(dir):
            img_path = os.path.join(dir, img_name)
            print(img_path)
            lst = my_utils.load_list(img_path)
            boxes = []
            for line in lst:
                box = {}
                split_line = line.split(" ")
                box["type"] = type
                if type == "gt":
                    box["class_id"] = split_line[0]
                    box["left"] = float(split_line[1])
                    box["top"] = float(split_line[2])
                    box["width"] = float(split_line[3])
                    box["height"] = float(split_line[4])
                else:
                    box["class_id"] = (split_line[0])  # class
                    box["confidence"] = float(split_line[1])
                    box["left"] = float(split_line[2])
                    box["top"] = float(split_line[3])
                    box["width"] = float(split_line[4])
                    box["height"] = float(split_line[5])

                boxes.append(box)

            if type == "gt":
                id2gt[img_name] = boxes
            else:
                id2det[img_name] = boxes

            # print("Type : {} - Image name : {} - Boxes : {}".format(type, img_name, boxes))

    print("Len id2gt : ", len(id2gt))
    print("Len id2det : ", len(id2det))

    evaluate(id2gt, id2det, iou_threshold=0.3)
