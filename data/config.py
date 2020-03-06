# config.py

cfg = {
    'name': 'FaceBoxes',
    #'min_dim': 1024,
    #'feature_maps': [[32, 32], [16, 16], [8, 8]],
    # 'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],     # default size of anchor box on inception_3, conv3_2, conv4_2
    'steps': [32, 64, 128],                         # tilling interval for default anchor
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}

mot_big_cfg = {
    'name': 'HumanBoxes_MOT',
    'rgb_mean': (112, 118, 120),
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'min_dim': 1024,
    'steps': [32, 64, 128],
    'densities': [[1], [1], [1]],
    'min_sizes': [[128], [256], [512]],
    'aspect_ratios': [[2.81], [2.81], [2.81]],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}

mot_normal_cfg = {
    'name': 'HumanBoxes_MOT',
    'rbg_mean': (112, 118, 120),
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'min_dim': 1024,
    'steps': [32, 64, 128],
    'densities': [[1], [1], [1]],
    'min_sizes': [[64], [128], [256]],
    'aspect_ratios': [[3], [3], [3]],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}

name2cfg = {
    "mot_big": mot_big_cfg,
    "mot_normal": mot_normal_cfg
}

def get_config(name):
    cfg = name2cfg.get(name)
    if cfg is None:
        raise "Config name [{}] is not valid !".format(name)
    else:
        return cfg