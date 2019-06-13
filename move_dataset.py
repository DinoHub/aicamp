import os
from shutil import copy

sources = ['/home/dh/Workspace/aicamp/data/TIL2019_v1.3/trainset_5classes_20406/train',
'/home/dh/Workspace/aicamp/data/TIL2019_v1.3/trainset_5classes_20406/val',
'/home/dh/Workspace/aicamp/data/TIL2019_v1.3/trainset_11classes_00000/train',
'/home/dh/Workspace/aicamp/data/TIL2019_v1.3/trainset_11classes_00000/val'
]

target = '/home/dh/Workspace/aicamp/data/TIL2019_v1.3_bodycrops/train'

for source in sources:
    print(source)
    for pose in os.listdir(source):
        pose_dir = os.path.join(source, pose)
        for img in os.listdir(pose_dir):
            if img.endswith('.png'):
                img_path = os.path.join(pose_dir, img)
                # print(img_path)
                tgt_path = os.path.join(target, pose)
                tgt_path = os.path.join(tgt_path, img)
                # print(tgt_path)
                copy(img_path, tgt_path)