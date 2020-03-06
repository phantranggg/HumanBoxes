from utils_dir import my_utils
import os
import argparse
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


sns.set()


def statistic_box_ratio_mot():
    dataset_dir = "/home/quanchu/Dataset/MOT17Det/train"
    dir_paths = my_utils.get_dir_paths(dataset_dir)

    col_names = ["FrameId", "ObjectId", "Left", "Top", "Width", "Height",
                 "IgnoreFlag", "ObjectClass", "VisibilityRatio"]
    df = []
    for dir_path in dir_paths:
        fpath = os.path.join(dir_path, "gt", "gt.txt")
        df.append(my_utils.load_csv(fpath, names=col_names))

    df = pd.concat(df)
    df = df[df["ObjectClass"].isin([1, 2, 7])]
    print(df.head())
    print(df.shape)
    # print(list(df.columns))

    df["box_ratio"] = df["Height"] / df["Width"]
    box_ratio_arr = df["box_ratio"].values
    min, max, avg, std = box_ratio_arr.min(), box_ratio_arr.max(), box_ratio_arr.mean(), box_ratio_arr.std()
    print("Min = {:.2f} - Max = {:.2f}".format(min, max))
    print("Avg = {:.2f} - Std = {:.2f}".format(avg, std))

    df = pd.DataFrame()
    print(box_ratio_arr.shape)
    x = box_ratio_arr
    y = np.ones(x.shape)
    # np.histogram2d(x, y, range=np.arange(min, max, 0.1))
    plt.hist(box_ratio_arr, bins=np.arange(min, max, 0.2))
    plt.xticks(np.arange(min, max, 0.5))

    plt.show()


def statistic_box_ratio():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='path of ground truth file')
    parser.add_argument('--dataset', default="mot", help='dataset name')
    args = parser.parse_args()
    gt_path = args.gt

    gt_json = my_utils.load_json(gt_path)
    num_imgs = len(gt_json)
    num_boxes = 0
    widths, heights = [], []
    for img_id, gt_boxes in gt_json.items():
        num_boxes += len(gt_boxes)
        for bb in gt_boxes:
            widths.append(bb["xmax"] - bb["xmin"] + 1)
            heights.append(bb["ymax"] - bb["ymin"] + 1)
            # total_width += bb["xmax"] - bb["xmin"] + 1
            # total_height += bb["ymax"] - bb["ymin"] + 1

    widths = np.array(widths)
    heights = np.array(heights)
    ratios = heights / widths
    # total_width, total_height = widths.sum(), heights.sum()

    print("\nStatistic on ground truth ", gt_path)
    print("Number images : ", num_imgs)
    print("Number boxes  : ", num_boxes)
    print("Average boxes/image          : {:.2f}".format(num_boxes / num_imgs))
    # print("Average width boxes          : {:.2f}".format(total_width / num_boxes))
    # print("Average height boxes         : {:.2f}".format(total_height / num_boxes))
    # print("Average ratio (height/width) : {:.2f}".format(total_height / total_width))

    for name, arr in [("Ratio", ratios), ("Width", widths), ("Height", heights)]:

        print("\nAverage {} : {:.2f}".format(name, arr.mean()))
        q = 0.5
        print("{:.2f} - Quantile : {:.2f}".format(q, np.quantile(arr, q)))

        if name == "Ratio":
            step = 0.2
        else:
            step = int(arr.std() / 20) * 10

        min_range = int(math.floor(arr.min()))
        max_range = int(math.ceil(arr.max() / step)) * step
        counts, bins = np.histogram(arr, bins=np.arange(min_range, max_range, step))
        counts = counts / counts.sum()
        # print(counts)
        # print(bins)
        x = (bins[:-1] + bins[1:]) / 2

        fig, ax = plt.subplots()
        plt.bar(x, counts, edgecolor="blue", width=step)
        plt.xticks(bins)
        plt.xlabel(name)
        plt.ylabel("Frequency")

        gcf = plt.gcf()
        gcf.set_size_inches(15, 8)
        save_path = "./eda/fig/{}/{}_{}.jpg".format(args.dataset, name, args.dataset)
        my_utils.make_parent_dirs(save_path)
        gcf.savefig(save_path, dpi=300)
        print("Save figure to {} done".format(save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='path of ground truth file')
    parser.add_argument('--dataset', default="mot", help='dataset name')
    args = parser.parse_args()
    gt_path = args.gt

    gt_json = my_utils.load_json(gt_path)
    num_imgs = len(gt_json)
    num_boxes = 0
    map_class_info = {}
    for img_id, gt_boxes in gt_json.items():
        num_boxes += len(gt_boxes)

        for bb in gt_boxes:
            class_id = bb["class_id"]
            info_of_class = map_class_info.get(class_id)
            if info_of_class is None:
                info_of_class = dict(widths=[], heights=[], num_boxes=0)
                map_class_info[class_id] = info_of_class

            info_of_class["widths"].append(bb["xmax"] - bb["xmin"] + 1)
            info_of_class["heights"].append(bb["ymax"] - bb["ymin"] + 1)
            info_of_class["num_boxes"] += 1



    # total_width, total_height = widths.sum(), heights.sum()

    print("\nStatistic on ground truth ", gt_path)
    print("Number images : ", num_imgs)
    print("Number boxes  : ", num_boxes)
    print("Average boxes/image          : {:.2f}".format(num_boxes / num_imgs))

    for class_id, info in map_class_info.items():
        widths = np.array(info["widths"])
        heights = np.array(info["heights"])
        num_boxes = info["num_boxes"]
        ratios = heights / widths
        print("\nClass {} : {} boxes".format(class_id, num_boxes))
        for name, arr in [("Ratio", ratios), ("Width", widths), ("Height", heights)]:

            print("Average {} : {:.2f}".format(name, arr.mean()))
            q = 0.5
            print("{:.2f} - Quantile {} : {:.2f}".format(q, name, np.quantile(arr, q)))

            # if name == "Ratio":
            #     step = 0.2
            # else:
            #     step = int(arr.std() / 20) * 10
            #
            # min_range = int(math.floor(arr.min()))
            # max_range = int(math.ceil(arr.max() / step)) * step
            # counts, bins = np.histogram(arr, bins=np.arange(min_range, max_range, step))
            # counts = counts / counts.sum()
            # # print(counts)
            # # print(bins)
            # x = (bins[:-1] + bins[1:]) / 2
            #
            # fig, ax = plt.subplots()
            # plt.bar(x, counts, edgecolor="blue", width=step)
            # plt.xticks(bins)
            # plt.xlabel(name)
            # plt.ylabel("Frequency")
            #
            # gcf = plt.gcf()
            # gcf.set_size_inches(15, 8)
            # save_path = "./eda/fig/{}/{}_{}.jpg".format(args.dataset, name, args.dataset)
            # my_utils.make_parent_dirs(save_path)
            # gcf.savefig(save_path, dpi=300)
            # print("Save figure to {} done".format(save_path))



if __name__ == "__main__":
    # statistic_box_ratio()
    main()