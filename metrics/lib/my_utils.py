import shutil
import os
from datetime import datetime
import pandas as pd

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_dir_paths(parent_dir):
    dir_paths = [os.path.join(parent_dir, dir) for dir in os.listdir(parent_dir)]
    dir_paths = [dir for dir in dir_paths if os.path.isdir(dir)]
    return dir_paths


def get_dir_names(parent_dir):
    dir_names = [dir_name for dir_name in os.listdir(parent_dir)
                 if os.path.isdir(os.path.join(parent_dir, dir_name))]
    return dir_names


def get_file_names(parent_dir):
    file_names = [file_name for file_name in os.listdir(parent_dir)
                  if os.path.isfile(os.path.join(parent_dir, file_name))]
    return file_names


def get_parent_path(path):
    return path[:path.rfind("/")]


def get_all_file_paths(dir, abs_path=False):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            if abs_path:
                path = os.path.abspath(path)
            file_paths.append(path)

    return file_paths


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_parent_dirs(path):
    dir = get_parent_path(path)
    make_dirs(dir)


def load_str(path):
    data = ""
    try:
        with open(path, 'r') as f:
            data = f.read().strip()
    except:
        print("Error when load str from ", os.path.abspath(path))

    return data


def save_str(data, save_path):
    make_parent_dirs(save_path)
    try:
        with open(save_path, 'w') as f:
            f.write(data)
        print("Save str data to {} done".format(save_path))
    except:
        print("Error when save str to ", os.path.abspath(save_path))


def get_time_str(time=datetime.now(), fmt=DEFAULT_TIME_FORMAT):
    try:
        return time.strftime(fmt)
    except:
        return ""


def save_list(lst, save_path):
    make_parent_dirs(save_path)

    with open(save_path, "w") as f:
        f.write("\n".join(lst))

    print("Save data (size = {}) to {} done".format(len(lst), save_path))


def load_list(path):
    data = []
    with open(path, 'r') as f:
        data = f.read().strip().split("\n")

    print("Load list data (size = {}) from {} done".format(len(data), path))
    return data


def load_csv(path, **kwargs):
    data = None
    try:
        data = pd.read_csv(path, **kwargs)
        print("Read csv data (size = {}) from {} done".format(data.shape[0], path))
    except:
        print("Error when load csv data from ", path)
    return data


def copy_file(src_path, dst_path):
    try:
        make_parent_dirs(dst_path)
        shutil.copyfile(src_path, dst_path)
        # print("Copy file from {} to {} done".format(src_path, dst_path))
        return True
    except Exception:
        print("Error: when copy file from {} to {}".format(src_path, dst_path))
        return False


def copy_files(src_dst_paths):
    total_paths = len(src_dst_paths)
    num_success = 0
    for i, (src_path, dst_path) in enumerate(src_dst_paths):
        if (i+1) % 10 == 0:
            print("Copying {}/{} ...".format(i+1, total_paths))
        is_success = copy_file(src_path, dst_path)
        if is_success:
            num_success += 1

    print("Copy {}/{} files done".format(num_success, total_paths))
    return num_success
