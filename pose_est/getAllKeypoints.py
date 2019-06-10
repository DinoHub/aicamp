# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import csv 

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('openpose/build/python/');
        # sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


def body_expanse(keypoints):
    xs = keypoints[:,0]
    ys = keypoints[:,1]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def best_body_idx(bodies_keypoints, thresh = 0.1):
    areas = []
    for body in bodies_keypoints:
        overall_conf = max(body[:,2])
        kps = np.array([kp[:2] for kp in body if kp[2] > thresh])
        # print('CHOICES',kps)
        if overall_conf > thresh:
            areas.append(body_expanse(kps))
        else:
            areas.append(0)
    # print('AREAS:',areas)
    return np.argmax(areas)

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
# params["model_folder"] = "../../../models/"
params["model_folder"] = "openpose/models/"
# params["face"] = True
params["hand"] = True

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = []
    for root, _, files in os.walk(args[0].image_dir, topdown=False):
        for name in files:
            if name.endswith('.png'):
                imagePaths.append(os.path.join(root, name))
    # imagePaths = op.get_images_on_directory(args[0].image_dir);
    print('TOTAL IMAGES IN ALL: ',len(imagePaths))
    start = time.time()
    

    csv_name = os.path.basename(os.path.dirname(root)) + '_keypoints.csv'
    csv_file = open(csv_name,mode='w') 
    headers = ['image_name', 'classname','image_width','image_height']

    num_keypoints = {'body':25, 'lefthand':21, 'righthand':21 }
    for cat in ['body','lefthand','righthand']:
        for keypoint_idx in range(num_keypoints[cat]):
            for field in ['x','y','conf']:
                headers.append('{}_kp{}_{}'.format(cat, keypoint_idx, field))

    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(headers)

    # Process and display images
    for imagePath in imagePaths:
        print(imagePath)
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        img_height, img_width = imageToProcess.shape[:2]
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        bodyKP = datum.poseKeypoints
        lhandKP = datum.handKeypoints[0]
        rhandKP = datum.handKeypoints[1]

        if bodyKP.shape:
            if bodyKP.shape[0] > 1:
                #more than 1
                best_idx = best_body_idx(bodyKP)
                bodyKP = bodyKP[best_idx]
                lhandKP = lhandKP[best_idx]
                rhandKP = rhandKP[best_idx]
                # print('CHOSEN',bodyKP)
                # cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
                # key = cv2.waitKey(0)
            else:
                # best_body(bodyKP)
                bodyKP = np.squeeze(bodyKP)
                lhandKP = np.squeeze(lhandKP)
                rhandKP = np.squeeze(rhandKP)
        else:
            # nobody
            # print('NOBODY')
            bodyKP = np.zeros((25,3))
            lhandKP = np.zeros((21,3))
            rhandKP = np.zeros((21,3))

        assert bodyKP.shape == (25,3),'body shape wrong'
        assert lhandKP.shape == (21,3),'lhand shape wrong {}'.format(lhandKP.shape)
        assert rhandKP.shape == (21,3),'rhand shape wrong {}'.format(rhandKP.shape)

        imagename = os.path.basename(imagePath)
        classname = imagename.split('_')[0]
        row = [imagename, classname, img_width, img_height]
        row.extend(bodyKP.flatten())
        row.extend(lhandKP.flatten())
        row.extend(rhandKP.flatten())
        csv_writer.writerow(row)

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    csv_file.close()
except Exception as e:
    print(e)
    sys.exit(-1)
