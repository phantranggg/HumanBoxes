## Convert MOT->VOC & Preprocessing
cd data/mot2voc
python MOT_to_VOC.py --mot_path ../MOT17Det/ --voc_path ../MOT17_voc_small/

## Visualize MOT with bbox
cd data/mot2voc
python MOT_VOC_visualization.py --voc_path ../MOT17_voc/train/MOT17-10

## Generate file img_list.txt for training (img_list.txt format: MOT17-05/img1/000009.jpg MOT17-05/gt/000009.xml)
cd data/mot2voc
python gen_list_img_voc.py

## Train model
python train.py --config_name mot_normal

## Continue training
python train.py --resume_epoch 300 --max_epoch 600 --resume_net ./mot_weights/Final_FaceBoxes.pth

## Generate img_list for test (input - a folder of images)
python gen_img_list.py --image_folder './data/MOT17_voc/test/MOT17-08/img1/' --save_folder './data/MOT17_voc/test/MOT17-08/'

## Convert video to image
python img2video.py --img_folder ./data/MOT17_voc/test/MOT17-03/img1 --save_dir ./data/MOT17_voc/test/MOT17-03/

## Test
python test.py --trained_model ./weights/Final_FaceBoxes.pth --save_folder ./MOT17_voc/test/res --show_image
python test.py --trained_model ./weights/Final_FaceBoxes.pth --dataset ./data/MOT17_voc/test/MOT17-06/ --save_folder ./data/MOT17_voc/test/MOT17-06/ --show_image --vis_thres 0.5 --nms_threshold 0.3 --top_k 100