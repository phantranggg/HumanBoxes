import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
cv2.setNumThreads(0)
from PIL import Image
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


MOT_CLASSES_INDEX = {
    "__background__": 0,
    "pedestrian": 1,
    "person on vehicle": 1,
    "static person": 1,
}

# CLASSES_INDEX_MOT_1 = {
#     "__background__": 0,
#     "human": 1,
#     "artificial": 2,
#     "occluder": 3,
#     "person on vehicle": 4,
#     "static person": 5,
# }

# CLASSES_INDEX_MOT_2 = {
#     "__background__": 0,
#     "human": 1,
#     "person on vehicle": 2,
#     "static person": 3,
#     "distractor": 4,
#     "reflection": 5,
#     "occluder": 6,
# }

# CLASSES_INDEX_CPS_2 = {
#     "__background__": 0,
#     "human": 1,
#     "rider": 2,
#     "sitting person": 3,
#     "other person": 4,
#     "group of people": 5,
# }

# CLASSES_INDEX_MOT_CPS = {
#     "__background__": 0,
#     "human": 1,
#     "rider": 2,
#     "static person": 3,
# }


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind="mot", keep_difficult=True):
        if class_to_ind == "mot":
            class_to_ind = MOT_CLASSES_INDEX
        # if class_to_ind == "mot_1":
        #     class_to_ind = CLASSES_INDEX_MOT_1
        # elif class_to_ind == "mot_2":
        #     class_to_ind = CLASSES_INDEX_MOT_2
        # elif class_to_ind == "cps_2":
        #     class_to_ind = CLASSES_INDEX_CPS_2
        # elif class_to_ind == "mot_cps":
        #     class_to_ind = CLASSES_INDEX_MOT_CPS
        else:
            raise "Class to index : {} is not valid".format(class_to_ind)

        self.class_to_ind = class_to_ind
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter('object'):
            # difficult = int(obj.find('difficult').text) == 1
            # if not self.keep_difficult and difficult:
            #     continue
            # name = obj.find('name').text.lower().strip()
            # if self.class_to_ind.get(name) is None:
            #     print("Class {} not valid".format(name))
            #     raise Exception

            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
            # label_idx = self.class_to_ind[name]
            label_idx = int(obj.find('label').text)
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res


class VOCDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, preproc=None, target_transform=None):
        self.root = os.path.join(root, 'train')
        self.preproc = preproc
        self.target_transform = target_transform
        # self._annopath = os.path.join(self.root, 'annotations', '%s')
        # self._imgpath = os.path.join(self.root, 'images', '%s')
        self._annopath = os.path.join(self.root, '%s')
        self._imgpath = os.path.join(self.root, '%s')
        self.ids = list()
        with open(os.path.join(self.root, 'img_list.txt'), 'r') as f:
          self.ids = [tuple(line.split()) for line in f]

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id[1]).getroot()
        img = cv2.imread(self._imgpath % img_id[0], cv2.IMREAD_COLOR)
        # img = Image.open(self._imgpath % img_id[0])
        if img is None:
            print("Error when load img from ", self._imgpath % img_id[0])
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)
            # print("Object num: ", target.shape)

        if self.preproc is not None:
            img, target = self.preproc(img, target)
            # print("Image shape & ground truth shape: ", img.shape, target.shape)

        return torch.from_numpy(img), target

    def __len__(self):
        return len(self.ids)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
