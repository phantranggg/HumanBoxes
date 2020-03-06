import os
import json
import pandas as pd
import numpy as np
import cv2
import argparse

import sys
sys.path.insert(1, '../')
from utils import my_utils

id2name = {
    1: "Pedestrian",
    2: "Person on vehicle",
    3: "Car",
    4: "Bicycle",
    5: "Motorbike",
    6: "Non motorized vehicle",
    7: "Static person",
    8: "Distractor",
    9: "Occluder",
    10: "Occluder on the ground",
    11: "Occluder full",
    12: "Reflection",
}

id2color = {
    1: (141, 97, 31),
    2: (138, 148, 241),
    3: (152, 60, 125),
    4: (222, 180, 210),
    5: (213, 179, 127),
    6: (255, 255, 255),
    7: (41, 57, 192),
    8: (176, 201, 72),
    9: (84, 153, 34),
    10: (15, 196, 241),
    11: (0, 84, 211),
    12: (164, 163, 153),
    -1: (0, 0, 0),
}


def statistic_mot_train_dataset(train_dir):
    mot_dirs = my_utils.get_dir_names(train_dir)
    col_names = ["FrameId", "ObjectId", "Left", "Top", "Width", "Height",
                 "IgnoreFlag", "ObjectClass", "VisibilityRatio"]

    result = []
    mot_dirs.sort(key=lambda x: int(x.split("-")[1]))
    for mot_dir in mot_dirs:
        gt_path = os.path.join(train_dir, mot_dir, "gt/gt.txt")

        df = my_utils.load_csv(gt_path, names=col_names)
        statistic_df = df["ObjectClass"].value_counts()
        statistic = statistic_df.to_dict()
        row = [statistic.get(i, 0) for i in range(1, 13)]
        row.append(np.sum(row))
        result.append(row)

    result.append(np.sum(result, axis=0).tolist())
    columns = [id2name.get(i) for i in range(1, 13)]
    columns.append("Total")
    result_df = pd.DataFrame(result, columns=columns)
    seqs = mot_dirs.copy()
    seqs.append("Total")
    result_df.insert(0, "Sequence", seqs)
    print(result_df)
    save_path = os.path.join("./result/statistic_mot_train.csv")
    my_utils.save_csv(result_df, save_path)


def is_select(curr_obj_id, boxes, coordinates):
    curr_left, curr_top, curr_width, curr_height = coordinates
    square = curr_width * curr_height
    is_select = True
    for bb in boxes:
        left, top, width, height = bb["left"], bb["top"], bb["width"], bb["height"]
        obj_id, class_id = bb["obj_id"], bb["class_id"]

        if obj_id == curr_obj_id or class_id not in [9, 10, 11]:
            continue

        max_left = max(curr_left, left)
        min_right = min(curr_left + curr_width - 1, left, + width - 1)
        max_top = max(curr_top, top)
        min_bottom = min(curr_top + curr_height - 1, top + height - 1)

        if max_left > min_right or max_top > min_bottom:
            continue

        overlap_square = (min_right - max_left + 1) * (min_bottom - max_top + 1)
        occlude_ratio = overlap_square / square

        if occlude_ratio >= 0.5:
            is_select = False
            break

    return is_select


def generate_bounding_boxes(img, boxes):
    remove_obj_ids = []
    for bb in boxes:
        left, top, width, height, class_id = bb["left"], bb["top"], bb["width"], \
                                                        bb["height"], bb["class_id"]
        obj_id, vis_ratio, ignore_flag = bb["obj_id"], bb["vis_ratio"], bb["ignore_flag"]
        left, top, width, height, class_id = int(left), int(top), int(width), \
                                                        int(height), int(class_id)
        obj_id, vis_ratio, ignore_flag = int(obj_id), float(vis_ratio), int(ignore_flag)
        # color = id2color[class_id]
        coordinates = (left, top, width, height)
        # if class_id in [1, 2, 7]:
        #     is_sel = is_select(obj_id, boxes, coordinates)
        # else:
        #     is_sel = True

        if ignore_flag == 0:
            color = id2color[-1]
            remove_obj_ids.append(obj_id)
        else:
            color = id2color[class_id]
        cv2.rectangle(img, pt1=(left, top), pt2=(left + width - 1, top + height - 1), color=color, thickness=2)

        text = "{}".format(obj_id)
        cv2.putText(img, text, (left + 5, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img, remove_obj_ids


def visualize_bounding_box(train_dir, save_dir):
    mot_dirs = my_utils.get_dir_names(train_dir)
    col_names = ["FrameId", "ObjectId", "Left", "Top", "Width", "Height",
                 "IgnoreFlag", "ObjectClass", "VisibilityRatio"]

    mot_dirs.sort(key=lambda x: int(x.split("-")[1]))
    remove_id_map = {}
    for mot_id, mot_dir in enumerate(mot_dirs):
        gt_path = os.path.join(train_dir, mot_dir, "gt/gt.txt")
        df = my_utils.load_csv(gt_path, names=col_names)
        base_img_dir = os.path.join(train_dir, mot_dir, "img1")
        img_names = my_utils.get_file_names(base_img_dir)
        remove_obj_id_of_mot_dir_map = {}

        for img_id, img_name in enumerate(img_names):
            frame_id = int(img_name[:img_name.rfind(".")])
            bboxes = []
            for _, row in df[df["FrameId"] == frame_id].iterrows():
                obj_id, object_class_id, left, top, width, height = row["ObjectId"], row["ObjectClass"], \
                                                                    row["Left"], row["Top"], \
                                                                    row["Width"], row["Height"]
                vis_ratio, ignore_flag = row["VisibilityRatio"], row["IgnoreFlag"]
                bboxes.append({
                    "obj_id": int(obj_id),
                    "ignore_flag": int(ignore_flag),
                    "vis_ratio": float(vis_ratio),
                    "class_id": int(object_class_id),
                    "left": float(left),
                    "top": float(top),
                    "width": float(width),
                    "height": float(height),
                })

            img = my_utils.load_img(os.path.join(base_img_dir, img_name))
            img, remove_obj_id_of_img = generate_bounding_boxes(img, bboxes)

            if len(remove_obj_id_of_img) > 0:
                remove_obj_id_of_mot_dir_map["frame_id"] = remove_obj_id_of_img

            save_img_path = os.path.join(save_dir, mot_dir, img_name)
            my_utils.save_img(img, save_img_path)

            if (img_id + 1) % 100 == 0:
                print("Processing {}/{} MOTs - {}/{} images done".format(
                    mot_id + 1, len(mot_dirs), img_id + 1, len(img_names)))

        if len(remove_obj_id_of_mot_dir_map) > 0:
            remove_id_map[mot_dir] = remove_obj_id_of_mot_dir_map

        if mot_id > 1:
            break

    save_remove_list_path = os.path.join(save_dir, "remove_list.txt")
    my_utils.save_json(remove_id_map, save_remove_list_path)
    for mot_dir, remove_map in remove_id_map.items():
        num_objects = 0
        for frame_id, remove_list in remove_map.items():
            num_objects += len(remove_list)
        print("{}: remove {} objects".format(mot_dir, num_objects))


if __name__ == "__main__":
    train_dir = "../data/MOT17Det/train/"
    save_dir = "../data/MOT_eda/"
    visualize_bounding_box(train_dir, save_dir)
