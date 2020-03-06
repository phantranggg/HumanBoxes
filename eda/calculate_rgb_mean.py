from utils_dir import my_utils
import cv2
import os
import time
import argparse
import numpy as np
import random


def calculate_rgb_mean(img_paths):
    start_time = time.time()
    num_err_files = 0
    imgs = []
    print("\nStart process {} images ...".format(len(img_paths)))
    for i, img_path in enumerate(img_paths):
        if (i + 1) % 100 == 0:
            print("Read {}/{} images ...".format(i + 1, len(img_paths)))
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_shape = img.shape
            # if i == 0:
            #     print("Image shape : ", img_shape)
            img = np.sum(img, axis=0).sum(axis=0)
            img = img / (img_shape[0] * img_shape[1])
            imgs.append(img)
        except:
            num_err_files += 1
    rgb_mean = np.mean(np.array(imgs), axis=0)
    exec_time = time.time() - start_time

    print("Num error files : ", num_err_files)
    print("Time : {:.2f} s".format(exec_time))
    print("RGB mean : ", rgb_mean)

    return rgb_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder contain images to calculate rgb mean", required=True)
    # parser.add_argument("--ext", default="jpg", help="Extension of image file, example jpg")
    parser.add_argument("--type", default="list", choices=["img", "list"], help="type = img (folder contain img file), "
                                                                                "type = list (folder contain file img list")
    args = parser.parse_args()

    if args.type == "img":
        img_paths = my_utils.get_all_file_paths(args.folder)
        # img_paths = [img_path for img_path in img_paths if img_path.endswith(args.ext)]
    else:
        img_list_path = os.path.join(args.folder, "img_list.txt")
        img_list = my_utils.load_list(img_list_path)
        img_list = [line.split(" ")[0] for line in img_list]

        img_paths = [os.path.join(args.folder, "images", line) for line in img_list]

    show_paths = random.sample(img_paths, 10)
    print(show_paths)
    rgb_mean = calculate_rgb_mean(img_paths)
