import os
import cv2

from yolo34py.pyod import Yolo34PyOD as ObjDetector

if __name__ == '__main__':
    objs_lists = ['person']
    od_threshold = 0.3
    # with HiddenPrints(): 
    od = ObjDetector( det_classes = objs_lists, threshold = od_threshold)

    parent_folder = '/home/dh/Workspace/aicamp/data/TIL2019_v0.1'

    bypass_poses = ['ChestBump', 'HandShake']
    count = 0

    for split in os.listdir( parent_folder ):
        split_dir = os.path.join(parent_folder, split)
        for pose in os.listdir(split_dir):
            pose_dir = os.path.join(split_dir, pose)
            print(pose_dir)
            for im in os.listdir(pose_dir):
                im_fp = os.path.join( pose_dir, im )
                img = cv2.imread( im_fp )

                if pose not in bypass_poses:
                    dets = od.get_detections( img )
                    # get the largest detection
                    largest_det = None
                    for det in dets:
                        if largest_det is None:
                            largest_det = det
                        else:
                            detbb = det['tlwh']
                            detarea = detbb['w'] * detbb['h']
                            ldbb = largest_det['tlwh']
                            ldarea = ldbb['w'] * ldbb['h']
                            if detarea > ldarea:
                                largest_det = det

                    if largest_det is not None:
                        # crop image
                        min_x = largest_det['topleft']['x']
                        min_y = largest_det['topleft']['y']
                        max_x = largest_det['bottomright']['x']
                        max_y = largest_det['bottomright']['y']

                        img_show = img[min_y:max_y, min_x:max_x]
                    else:
                        # copy the image as per usual
                        img_show = img

                    print(dets)
                    cv2.imshow('', img_show)
                    cv2.waitKey(0)
                    count += 1
                    if count > 20:
                        exit()
                else:
                    print('copying the image exactly since bypass')
                    # TODO

    # TODO
    # 1. expand the bb a little bit
    # 2. copy over the folder structure of the parent, retaining the undetected and the bypassed

