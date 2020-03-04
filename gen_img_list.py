import os
import argparse

parser = argparse.ArgumentParser(description='Generate image list for testing')
parser.add_argument('--image_folder', default='', help='Name of image folder for testing')
parser.add_argument('--save_folder', default='', help='Name of folder to save img_list.txt file')
args = parser.parse_args()

f = open(os.path.join(args.save_folder, 'img_list.txt'), 'w', newline='')
for data_file in sorted(os.listdir(args.image_folder)):
    # print(data_file[:-4])
    f.write(data_file[:-4] + "\n")
f.close()
print("Done!")