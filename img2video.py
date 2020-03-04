import cv2, os
import numpy as np
import argparse
from utils.my_utils import get_all_file_paths_in_order

parser = argparse.ArgumentParser(description="Image to Video Converter")
parser.add_argument('--img_folder', default=None, help='Input images folder')
parser.add_argument('--save_dir', default=None, help='Dir to save result video')
args = parser.parse_args()

size = (1920, 1080) # (w, h)
out = cv2.VideoWriter(os.path.join(args.save_dir, 'input.mp4'), cv2.VideoWriter_fourcc(*'MJPG'), 15, size)

all_file_paths = get_all_file_paths_in_order(args.img_folder)
print(len(all_file_paths))
# print(all_file_paths[0:10])
for filename in all_file_paths:
    # print(filename)
    img = cv2.imread(filename)
    out.write(img)

print("Done!")
out.release()