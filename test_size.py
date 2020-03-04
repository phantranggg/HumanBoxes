import numpy as np

data_dir = './data/WiderPerson/Images/'
annotation = './data/WiderPerson/Annotations/'
train_file = './data/WiderPerson/train.txt'

res = np.empty((0, 5))
f = open(train_file, "r")
train_set = []
for filename in f:
    train_set.append(filename[:-1])
f.close()
print("Train set: ", len(train_set))



for filename in train_set:
    try:
        f = open(annotation + filename + ".jpg.txt", "r")
        num_bboxes = f.readline()
        for s in f:
            label_bbox = s[:-1].split(' ')
            label, bbox = int(label_bbox[0]), np.array(label_bbox[1:], dtype=np.int64)
            res = np.vstack((res, np.append(bbox, label))) # [xmin, ymin, xmax, ymax]
        print(res.shape)
        # return res

    except FileNotFoundError:
        print("file not found")
    