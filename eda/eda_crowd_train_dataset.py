import os
import time
import cv2
import random

import sys
sys.path.insert(1, '../')
from utils import my_utils

class2color = {
    "person": (141, 97, 31),
    "mask": (0, 84, 211),
    -1: (0, 0, 0),
}


def generate_bounding_boxes(img, boxes):
    for bb in boxes:
        obj_class, fbox = bb["tag"], bb["fbox"]
        left, top, width, height = fbox
        left, top, width, height = int(left), int(top), int(width), int(height)

        occ = None
        try:
            occ = float(bb["extra"]["occ"])
        except:
            occ = None

        color = class2color[obj_class]
        r = random.random()
        threshold = 0.1
        if obj_class == "mask" or (obj_class == "person" and r < threshold):
            cv2.rectangle(img, pt1=(left, top), pt2=(left + width, top + height), color=color, thickness=2)

        if obj_class == "mask" or (r < threshold):
            if occ is not None:
                text = "{:.2f}".format(occ)
                cv2.putText(img, text, (left + 5, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


if __name__ == "__main__":
    start_time = time.time()

    src_dir = "../data/CrowdHuman/"
    dst_dir = "../data/CrowdHuman/train01_visualize"
    train_dir = os.path.join(src_dir, "train01")
    src_img_names = my_utils.get_file_names(train_dir)

    org_annotation_path = os.path.join(src_dir, "annotation_train.odgt")
    org_annotation = my_utils.load_json_lines(org_annotation_path)
    org_annotation = {elm["ID"]: elm["gtboxes"] for elm in org_annotation}

    num_sel_images = 0
    for i, src_img_name in enumerate(src_img_names):
        key_img_name = src_img_name[:src_img_name.rfind(".")]
        boxes = org_annotation[key_img_name]
        num_persons, total_width = 0, 0
        for bb in boxes:
            obj_class, fbox = bb["tag"], bb["fbox"]
            left, top, width, height = fbox
            left, top, width, height = int(left), int(top), int(width), int(height)

            if obj_class == "person":
                total_width += width
                num_persons += 1

        if num_persons >= 10 and (total_width < 80 * num_persons):

            img = my_utils.load_img(os.path.join(train_dir, src_img_name))
            img = generate_bounding_boxes(img, boxes)

            dst_img_path = os.path.join(dst_dir, src_img_name)
            my_utils.save_img(img, dst_img_path)

            num_sel_images += 1

        if num_sel_images > 100:
            exit()

        if (i + 1) % 100 == 0:
            print("Processing {}/{} images done".format(i + 1, len(src_img_names)))

    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))
