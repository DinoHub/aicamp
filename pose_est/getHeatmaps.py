# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

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

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
# parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "openpose/models/"
# params["model_folder"] = "../../../models/"
# params["hand"] = True
params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2

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

# Read frames on directory
imagePaths = []
for root, _, files in os.walk(args[0].image_dir, topdown=False):
    for name in files:
        if name.endswith('.png'):
            imagePaths.append(os.path.join(root, name))
# imagePaths = op.get_images_on_directory(args[0].image_dir);
print('TOTAL IMAGES IN ALL: ',len(imagePaths))


try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    for imagePath in imagePaths:
        print(imagePath)        
        imagename = os.path.basename(imagePath)
        imagename = imagename.split('.')[0]
        # classname = imagename.split('_')[0]
        # print(imagename, classname)

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        img_height, img_width = imageToProcess.shape[:2]
        # print(img_height, img_width)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # Process outputs
        outputImageF = (datum.inputNetData[0].copy())[0,:,:,:] + 0.5
        outputImageF = cv2.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
        outputImageF = (outputImageF*255.).astype(dtype='uint8')

        pose_heatmaps = datum.poseHeatMaps.copy()
        pose_heatmaps = (pose_heatmaps).astype(dtype='uint8')

        pose_heatmaps = np.moveaxis(pose_heatmaps, 0, -1)
        # print(pose_heatmaps.shape)

        # ( height, width, channels(rgb+heatmaps) )
        posehm_and_rgb = np.concatenate((outputImageF, pose_heatmaps), axis=2)
        # print(posehm_and_rgb.shape)

        np.save('heatmaps/{}'.format(imagename), posehm_and_rgb, allow_pickle=False)


    exit()

    # hand_heatmaps = datum.handHeatMaps.copy()
    # lhand_heatmaps = (hand_heatmaps[0][0]).astype(dtype='uint8')
    # rhand_heatmaps = (hand_heatmaps[1][0]).astype(dtype='uint8')


    # # print(img_height/img_width)
    # print(pose_heatmaps.shape)
    # print(lhand_heatmaps.shape)
    # print(rhand_heatmaps.shape)

    # lhand_heatmaps = np.array([cv2.resize(hm, outputImageF.shape[:2][::-1]) for hm in lhand_heatmaps])
    # rhand_heatmaps = np.array([cv2.resize(hm, outputImageF.shape[:2][::-1]) for hm in rhand_heatmaps])

    # print(lhand_heatmaps.shape)
    # print(rhand_heatmaps.shape)

    # Display Image
    counter = 0
    while 1:
        num_maps = pose_heatmaps.shape[0]
        heatmap = pose_heatmaps[counter, :, :].copy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", combined)
        cv2.imshow("OpenPose 1.5.0 - Tutorial Python API 2", imageToProcess)
        # cv2.imshow("OpenPose 1.5.0 - Tutorial Python API 3", lhand_heatmaps[counter%21])
        # cv2.imshow("OpenPose 1.5.0 - Tutorial Python API 4", rhand_heatmaps[counter%21])
        # cv
        key = cv2.waitKey(-1)
        if key == 27:
            break
        counter += 1
        counter = counter % num_maps
except Exception as e:
    print(e)
    sys.exit(-1)