import os
import cv2
import random
import shutil

def rename_to_consistent():
    parent_dir = 'competition'
    count = 0
    for pose in os.listdir( parent_dir ):
        pose_dirp = os.path.join( parent_dir, pose )
        for pid in os.listdir( pose_dirp ):
            pid_dirp = os.path.join( pose_dirp, pid )
            for img in os.listdir( pid_dirp ):
                src_fp = os.path.join( pid_dirp, img )
                dst_fp = os.path.join( pid_dirp, '{}_{}_{}.png'.format(pose, pid, count) )
                count += 1
                print('renaming {} -> {}'.format(src_fp, dst_fp))
                os.rename( src_fp, dst_fp )

def count_imgs( pose_dirp ):
    img_count = 0
    for _, _, files in os.walk( pose_dirp ):
        img_count += len( [f for f in files if f.endswith('.png')] )
    return img_count

def sample_a_competition():
    parent_dir = 'competition'
    target_dir = 'competition_split'
    test_target_ratio = 0.2
    for pose in os.listdir( parent_dir ):
        target_pose_dir = os.path.join( target_dir, 'test' )
        target_pose_dir = os.path.join( target_pose_dir, pose )
        if not os.path.exists( target_pose_dir ):
            os.makedirs( target_pose_dir )

        pose_dirp = os.path.join( parent_dir, pose )
        posewise_imgs = count_imgs(pose_dirp)
        target_test_count = int( posewise_imgs * test_target_ratio )

        shuff_list = list(os.listdir( pose_dirp ))
        random.shuffle( shuff_list )
        curr_count = 0
        for pid in shuff_list:
            if curr_count >= target_test_count:
                target_pose_dir = os.path.join( target_dir, 'train' )
                target_pose_dir = os.path.join( target_pose_dir, pose )
                if not os.path.exists( target_pose_dir ):
                    os.makedirs( target_pose_dir )

            pid_dirp = os.path.join( pose_dirp, pid )
            src_tgt_pairs = [(os.path.join(pid_dirp, f), os.path.join(target_pose_dir, f)) for f in os.listdir( pid_dirp ) if f.endswith('.png')]
            curr_count += len( src_tgt_pairs )
            # perform a write to target_pose_dir
            for src_fp, tgt_fp in src_tgt_pairs:
                shutil.copyfile( src_fp, tgt_fp )

from collections import defaultdict

def groupby_pids(pose_dirp):
    pose_dict = defaultdict(list)
    img_names = os.listdir( pose_dirp )
    for img_name in img_names:
        _, pid, _ = img_name.split('_')
        pose_dict[pid].append( img_name )
    return pose_dict

def sample_aicamp():
    parent_dir = 'P16SES'
    target_dir = 'TIL2019'
    # split_scheme = [(0.7,'train'), (0.1,'val'), (0.1,'publictest'), (0.1,'privatetest')]
    split_scheme = [(0.85,'train'), (0.15,'test')]
    for pose in os.listdir( parent_dir ):
        pose_dirp = os.path.join( parent_dir, pose )
        posewise_imgs = len(list(os.listdir(pose_dirp)))
        pose_dict = groupby_pids( pose_dirp )
        # shuff_list = list(os.listdir( pose_dirp ))
        shuff_list = list(pose_dict.keys())
        random.shuffle( shuff_list )

        idx = 0

        for ratio, split in split_scheme:
            target_pose_dir = os.path.join( target_dir, split )
            target_pose_dir = os.path.join( target_pose_dir, pose )
            if not os.path.exists( target_pose_dir ):
                os.makedirs( target_pose_dir )

            target_count = int( posewise_imgs * ratio )

            curr_count = 0
            while curr_count < target_count and idx < len(shuff_list):
            # for pid in shuff_list:
                # if curr_count >= target_test_count:
                #     target_pose_dir = os.path.join( target_dir, 'train' )
                #     target_pose_dir = os.path.join( target_pose_dir, pose )
                #     if not os.path.exists( target_pose_dir ):
                #         os.makedirs( target_pose_dir )
                pid = shuff_list[idx]

                # pid_dirp = os.path.join( pose_dirp, pid )
                src_tgt_pairs = [(os.path.join(pose_dirp, f), os.path.join(target_pose_dir, f)) for f in pose_dict[pid]]
                curr_count += len( src_tgt_pairs )
                # perform a write to target_pose_dir
                for src_fp, tgt_fp in src_tgt_pairs:
                    # print('copying {} to {}'.format(src_fp, tgt_fp))
                    shutil.copyfile( src_fp, tgt_fp )
                idx += 1
        while idx < len(shuff_list):
            pid = shuff_list[idx]

            pid_dirp = os.path.join( pose_dirp, pid )
            src_tgt_pairs = [(os.path.join(pose_dirp, f), os.path.join(target_pose_dir, f)) for f in pose_dict[pid]]
            # perform a write to target_pose_dir
            for src_fp, tgt_fp in src_tgt_pairs:
                # print('copying {} to {}'.format(src_fp, tgt_fp))
                shutil.copyfile( src_fp, tgt_fp )
            idx += 1

def sample_validation_set():
    train_folder = '/home/dh/Workspace/aicamp/data/TIL2019_v0.1/train'
    val_folder = '/home/dh/Workspace/aicamp/data/TIL2019_v0.1/val'
    ratio = 0.15

    for pose in os.listdir( train_folder ):
        pose_dirp = os.path.join( train_folder, pose )
        posewise_imgs = len(list(os.listdir(pose_dirp)))
        pose_dict = groupby_pids( pose_dirp )
        # shuff_list = list(os.listdir( pose_dirp ))
        shuff_list = list(pose_dict.keys())
        random.shuffle( shuff_list )

        idx = 0

        target_pose_dir = os.path.join( val_folder, pose )
        if not os.path.exists( target_pose_dir ):
            os.makedirs( target_pose_dir )

        target_count = int( posewise_imgs * ratio )

        curr_count = 0
        while curr_count < target_count and idx < len(shuff_list):
            pid = shuff_list[idx]

            src_tgt_pairs = [(os.path.join(pose_dirp, f), os.path.join(target_pose_dir, f)) for f in pose_dict[pid]]
            curr_count += len( src_tgt_pairs )
            # perform a write to target_pose_dir
            for src_fp, tgt_fp in src_tgt_pairs:
                # print('moving {} to {}'.format(src_fp, tgt_fp))
                shutil.move( src_fp, tgt_fp )
            idx += 1

if __name__ == '__main__':
    # sample_aicamp()
    sample_validation_set()
    # sample_a_competition()
    # rename_to_consistent()


