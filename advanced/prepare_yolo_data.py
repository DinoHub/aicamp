import os

from PIL import Image
from kerasyolo.yolo import YOLO

def yolo_poses_in_folder(pose_folder, target_folder, od):
    for pose in os.listdir(pose_folder):
        # pose_failcount = 0
        pose_dir = os.path.join(pose_folder, pose)
        tgt_pose_dir = os.path.join(target_folder, pose)
        if not os.path.exists(tgt_pose_dir):
            os.makedirs( tgt_pose_dir )

        for im in os.listdir(pose_dir):
            im_fp = os.path.join( pose_dir, im )
            target_fp = os.path.join( tgt_pose_dir, im )
            # img = cv2.imread( im_fp )

            # if pose not in bypass_poses:
            img = Image.open(im_fp)
            img_show = od.crop_largest_person( img )
            img_show.save( target_fp, os.path.splitext(target_fp)[1][1:].upper() )

def yolo_flat(flat_folder, target_parent_folder, od):
    target_folder = os.path.join( target_parent_folder, flat_folder )
    print('target_folder: {}'.format(target_folder))
    if not os.path.exists( target_folder ):
        os.makedirs( target_folder )
    for im in os.listdir(flat_folder):
        src = os.path.join( flat_folder, im )
        dst = os.path.join( target_folder, im )
        img = Image.open(src)
        img_show = od.crop_largest_person( img )
        img_show.save( dst, 'PNG' )


if __name__ == '__main__':
    objs_lists = ['person']
    od_threshold = 0.3
    # with HiddenPrints(): 
    # od = ObjDetector( det_classes = objs_lists, threshold = od_threshold)
    od = YOLO(threshold=od_threshold)

    # parent_folder = '/home/dh/Workspace/aicamp/data/TIL2019_v1.3/trainset_5classes_20406/train/ChestBump'
    target_folder = '/home/dh/Workspace/aicamp/data/cropped_data/TIL2019_v1.3'

    # 11classes = ['ChairPose','ChildPose','Dabbing', 'HandGun', 'HandShake', 'HulkSmash', 'KoreanHeart', 'KungfuCrane', 'KungfuSalute', 'Salute', 'WarriorPose']
    # 11classes = ['ChairPose','ChildPose']
    # 4classes = ['ChestBump', 'EaglePose', 'HighKneel', 'Spiderman']

    folders = ['/home/dh/Workspace/aicamp/data/TIL2019_v1.3/trainset_11classes_00000/val/{}'.format(s) for s in ['ChairPose','ChildPose','Dabbing', 'HandGun', 'HandShake', 'HulkSmash', 'KoreanHeart', 'KungfuCrane', 'KungfuSalute', 'Salute', 'WarriorPose'] ]
    for parent_folder in folders:
        yolo_flat( parent_folder, target_folder, od )

    exit()

    # count = 0

    # for pose in os.listdir( parent_folder ):
    #     pose_folder = os.path.join(parent_folder, pose)
    #     target_pose_folder = os.path.join(target_folder, pose)
    yolo_poses_in_folder( parent_folder, target_folder, od )
        # for pose in os.listdir(split_dir):
        #     pose_failcount = 0
        #     pose_dir = os.path.join(split_dir, pose)
        #     tgt_pose_dir = os.path.join(tgt_split_dir, pose)
        #     if not os.path.exists(tgt_pose_dir):
        #         os.makedirs( tgt_pose_dir )

        #     for im in os.listdir(pose_dir):
        #         im_fp = os.path.join( pose_dir, im )
        #         target_fp = os.path.join( tgt_pose_dir, im )
        #         # img = cv2.imread( im_fp )

        #         # if pose not in bypass_poses:
        #         img = Image.open(im_fp)
        #         img_show = od.crop_largest_person( img )
        #         img_show.save( target_fp, os.path.splitext(target_fp)[1][1:].upper() )
                # dets = od.detect_persons( img )
                # # get the largest detection
                # largest_det = None
                # for _,_,tlbr in dets:
                #     if largest_det is None:
                #         largest_det = tlbr
                #     else:
                #         detarea = (tlbr[3]-tlbr[1]) * (tlbr[2]-tlbr[0])
                #         ldarea = (largest_det[3]-largest_det[1]) * (largest_det[2]-largest_det[0])
                #         if detarea > ldarea:
                #             largest_det = tlbr

                # if largest_det is not None:
                #     # crop image
                #     min_x = largest_det[1]
                #     min_y = largest_det[0]
                #     max_x = largest_det[3]
                #     max_y = largest_det[2]

                #     # img_show = img[min_y:max_y, min_x:max_x]
                #     img_show = img.crop( (min_x, min_y, max_x, max_y) )
                # else:
                #     pose_failcount +=1
                #     # copy the image as per usual
                #     img_show = img
                # img_show.save( target_fp, os.path.splitext(target_fp)[1][1:].upper() )
            # pose_total = len(list(os.listdir(pose_dir)))
            # print('for pose {}: {} / {}'.format(pose, pose_failcount, pose_total))

                # print(dets)
                # cv2.imshow('', img_show)
                # cv2.waitKey(0)
                # img_show.show()
                # input()
                # count += 1
                # if count > 0:
                #     exit()
    # TODO
    # 1. expand the bb a little bit
    # 2. copy over the folder structure of the parent, retaining the undetected and the bypassed

