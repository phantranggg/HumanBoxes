from metrics.eval_cps.coco import COCO
from metrics.eval_cps.eval_MR_multisetup import COCOeval
import argparse
import os
import time
from datetime import datetime

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_time_str(time=datetime.now(), fmt=DEFAULT_TIME_FORMAT):
    try:
        return time.strftime(fmt)
    except:
        return ""


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_parent_path(path):
    return path[:path.rfind("/")]


def make_parent_dirs(path):
    dir = get_parent_path(path)
    make_dirs(dir)


def eval_cps(annFile, resFile, save_folder):
    annType = 'bbox'      #specify type here
    print('Running demo for {} results.'.format(annType))
    save_path = os.path.join(save_folder, "result_eval_cps.txt")
    make_parent_dirs(save_path)

    save_file = open(save_path, "w")
    for id_setup in range(0,4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        cocoEval.summarize(id_setup, save_file)

    save_file.close()


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='CityPerson Evaluation')
    parser.add_argument('--ann', required=True, help='annotation path')
    parser.add_argument('--res', required=True, help='detection path')
    parser.add_argument('--save_folder', required=True, help='save folder')

    args = parser.parse_args()
    annFile = args.ann
    resFile = args.res
    save_folder = args.save_folder
    eval_cps(annFile, resFile, save_folder)

    exec_time = time.time() - start_time
    print("Evaluation time : {:.2f}s".format(exec_time))
