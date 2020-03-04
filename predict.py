from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import get_config
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
import cv2
from utils.box_utils import decode
from utils.timer import Timer
from utils import my_utils
from metrics.eval_cps.eval_demo import eval_cps
import time
from utils.plot_utils import generate_bounding_boxes
from models import model_utils
import math

parser = argparse.ArgumentParser(description='HumanBoxes Testing')

parser.add_argument('--cfg', required=True, help='Config name, example: mot, crowd,...')
parser.add_argument('-m', '--trained_model', default='weights/HumanBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./data/predict/', type=str, help='Dir to save results')
parser.add_argument('--draw_boxes', action="store_true", help='Visualize boxes')
parser.add_argument('--predict_folder', type=str, help='Dir contain predict images', required=True)
parser.add_argument('--num_classes', default=2, type=int, help='num classes')

parser.add_argument('--vga_ratio', default=None, type=float, help='resize predict image to new size compare with vga resolution')
parser.add_argument('--vga_res', action="store_true", help='resize predict image to 640x480')
parser.add_argument('--min_width_input', type=int, default=0, help='convert input to satisfy min width')

parser.add_argument('--use_cpu', action="store_true", help='Use cpu to predict')
parser.add_argument('--use_nms_cpu', action="store_true", help='Use cpu nms')
parser.add_argument('--use_soft_nms', action="store_true", help='Use soft nms')
parser.add_argument('--confidence_threshold', default=0.4, type=float, help='confidence_threshold')
parser.add_argument('--min_height', default=0, type=int, help='min height of predicted box')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--ext', default="jpg,jpeg,png", help='Extensions of selected predict image files, example: jpg, png, ...')
args = parser.parse_args()

if args.use_soft_nms:
    args.use_nms_cpu = True


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(pretrained_path, use_cuda=True):
    print('Loading pretrained model from {} with using cuda : {}'.format(pretrained_path, use_cuda))

    if use_cuda:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    else:
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")

    if "state_dict" in pretrained_dict.keys():
        state_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        state_dict = remove_prefix(pretrained_dict, 'module.')

    model_name = pretrained_dict["model_name"]
    print("model_name: ", model_name)
    model_name = "HumanBoxes"
    model = model_utils.get_model_architecture(model_name, phase='test', size=None,
                                               num_classes=args.num_classes, use_cuda=use_cuda)
    check_keys(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    start_time = time.time()
    cfg = get_config(args.cfg)

    use_cuda = not args.use_cpu
    use_nms_cpu = args.use_nms_cpu
    use_soft_nms = args.use_soft_nms

    # net and model
    # net = HumanBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(args.trained_model, use_cuda)
    net.eval()
    print('Finished loading model!')
    # print(net)
    if use_cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    # Load predict image paths
    predict_folder = args.predict_folder
    predict_image_paths = my_utils.get_all_file_paths(predict_folder)
    extensions = args.ext.split(",")
    predict_image_paths = my_utils.get_files_with_extension(predict_image_paths, extensions)
    total_pred_images = len(predict_image_paths)

    # testing scale
    # if args.dataset == "FDDB":
    #     resize = 3
    # elif args.dataset == "PASCAL":
    #     resize = 2.5
    # elif args.dataset == "AFW":
    #     resize = 1
    resize = 1

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # predicting begin
    id2det = {}
    test_time, total_detect_time, total_nms_time = 0, 0, 0
    total_boxes = 0
    num_pred_images = 0
    error_image_paths = []
    for i, image_path in enumerate(predict_image_paths):
        image_path = os.path.abspath(image_path)
        image_name = os.path.basename(image_path)
        # if i < 2:
        #     print("Image_path : {} - Image name : {}".format(image_path, image_name))

        try:
            img = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))

        except:
            error_image_paths.append(image_path)
            continue
        old_h, old_w, _ = img.shape
        old_scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        if args.vga_res:
            img = cv2.resize(img, (640, 480), None, interpolation=cv2.INTER_LINEAR)
        elif args.vga_ratio:
            # old_h, old_w, _ = img.shape
            new_square = args.vga_ratio * args.vga_ratio * 640 * 480
            resize = math.sqrt(new_square / (old_h * old_w))
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        resize = 1
        if args.min_width_input > 0:
            while img.shape[1] * resize < args.min_width_input:
                resize += 0.5
            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        # if resize != 1:
        #     img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        new_im_height, new_im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= cfg["rgb_mean"]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        if use_cuda:
            img = img.cuda()
            scale = scale.cuda()
            old_scale = old_scale.cuda()

        with torch.no_grad():
            _t['forward_pass'].tic()
            out = net(img)  # forward pass
            detect_time = _t['forward_pass'].toc(average=False)

            _t['misc'].tic()
            priorbox = PriorBox(cfg, out[2], (new_im_height, new_im_width), phase='test')
            priors = priorbox.forward()
            if use_cuda:
                priors = priors.cuda()
            loc, conf, _ = out
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            # boxes = boxes * scale / resize
            boxes = boxes * old_scale
            boxes = boxes.cpu().numpy()
            # scores = conf.data.cpu().numpy()[:, 1]

            scores = conf.data.cpu().numpy()

            max_score_idx = np.argmax(scores, axis=1)
            sel_idx = max_score_idx == 1
            scores = scores[sel_idx, 1]
            boxes = boxes[sel_idx, :]

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # ignore low height box
            if args.min_height > 0:
                sel_idx = (boxes[:, 2] - boxes[:, 0] + 1) > args.min_height
                boxes = boxes[sel_idx]
                scores = scores[sel_idx]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(dets, args.nms_threshold, force_cpu=use_nms_cpu,
                       use_soft_nms=use_soft_nms, conf_threshold=args.confidence_threshold)
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]

            nms_time = _t['misc'].toc(average=False)

        det_bboxes = []
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            width = xmax - xmin + 1
            height = ymax - ymin + 1

            det_bboxes.append({
                "type": "det",
                "class_id": "human",
                "confidence": float(score),
                "left": float(xmin),
                "top": float(ymin),
                "width": float(width),
                "height": float(height),
            })

        id2det[image_path] = det_bboxes
        total_boxes += len(det_bboxes)
        num_pred_images += 1

        total_detect_time += detect_time
        total_nms_time += nms_time
        test_time = total_detect_time + total_nms_time
        fps = num_pred_images / test_time
        print('im_detect: {:d}/{:d} - forward_pass: {:.4f}s - nms: {:.4f}s - '
              'fps: {:.2f} - ({}x{}) -> ({}x{})'.format(
                i + 1, total_pred_images, detect_time, nms_time, fps, old_w, old_h, new_im_width, new_im_height))

    total_time = time.time() - start_time

    save_folder = os.path.join(args.save_folder, my_utils.get_time_str())
    my_utils.make_dirs(save_folder)
    save_pred_path = os.path.join(save_folder, "id2det.txt")
    my_utils.save_json(id2det, save_pred_path)

    if args.draw_boxes:
        print("\n\n Visualizing boxes ...")
        save_vis_folder = os.path.join(save_folder, "visualize")
        num_imgs = 0
        for img_path, bboxes in id2det.items():
            img_name = os.path.basename(img_path)
            img = my_utils.load_img(img_path)
            img = generate_bounding_boxes(img, bboxes)
            save_img_path = os.path.join(save_vis_folder, img_name)
            my_utils.save_img(img, save_img_path)

            num_imgs += 1
            if num_imgs % 100 == 0:
                print("Save {}/{} visualized images ...".format(num_imgs, len(id2det)))

    logs = []
    curr_time = my_utils.get_time_str()
    logs.append(curr_time)
    logs.append("\n\nArguments\n")
    for k, v in vars(args).items():
        logs.append("{} : {}".format(k, v))

    logs.append("\n\n")

    curr_log = "Total images: {}".format(total_pred_images)
    print(curr_log)
    logs.append(curr_log)

    curr_log = "Num error images: {}".format(len(error_image_paths))
    print(curr_log)
    logs.append(curr_log)

    if len(error_image_paths) > 0:
        curr_log = "Example error image paths: {}".format(error_image_paths[:5])
        print(curr_log)
        logs.append(curr_log)

    curr_log = "Num predicted images : {}".format(num_pred_images)
    print(curr_log)
    logs.append(curr_log)

    curr_log = "Num predicted boxes : {}".format(total_boxes)
    print(curr_log)
    logs.append(curr_log)

    test_time = total_detect_time + total_nms_time
    fps = num_pred_images / test_time
    curr_log = "\nAverage {:.4f} s/img. FPS : {:.2f} (vga_ratio = {})".format(1 / fps, fps, args.vga_ratio)
    print(curr_log)
    logs.append(curr_log)

    curr_log = "Predict time : {:.4f}s".format(test_time)
    print(curr_log)
    logs.append(curr_log)

    curr_log = "Detect time  : {:.4f}s ({:.2f}%)".format(total_detect_time, total_detect_time * 100 / test_time)
    print(curr_log)
    logs.append(curr_log)

    curr_log = "Nms time     : {:.4f}s ({:.2f}%)".format(total_nms_time, total_nms_time * 100 / test_time)
    print(curr_log)
    logs.append(curr_log)

    total_time = time.time() - start_time
    curr_log = "\nTotal time run script: {:.2f} seconds".format(total_time)
    print(curr_log)
    logs.append(curr_log)

    save_log_path = os.path.join(save_folder, "predict_log.txt")
    my_utils.save_list(logs, save_log_path)
