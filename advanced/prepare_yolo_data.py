import os

from PIL import Image
from kerasyolo.yolo import YOLO

if __name__ == '__main__':
    objs_lists = ['person']
    od_threshold = 0.3
    # with HiddenPrints(): 
    # od = ObjDetector( det_classes = objs_lists, threshold = od_threshold)
    od = YOLO(threshold=od_threshold)

    parent_folder = '/home/angeugn/Workspace/aicamp/data/TIL2019_v0.1'
    target_folder = '/home/angeugn/Workspace/aicamp/data/TIL2019_v0.1_yoloed'

    count = 0

    for split in os.listdir( parent_folder ):
        split_dir = os.path.join(parent_folder, split)
        tgt_split_dir = os.path.join(target_folder, split)
        for pose in os.listdir(split_dir):
            pose_failcount = 0
            pose_dir = os.path.join(split_dir, pose)
            tgt_pose_dir = os.path.join(tgt_split_dir, pose)
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

