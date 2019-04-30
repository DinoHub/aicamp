import argparse
import cv2
import time
import os 
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-vp','--vid_path', help='Video filepaths/streams for \
                    all cameras, e.g.: 0')
parser.add_argument('-out','--out_path', help='Output path to write images to')
args = parser.parse_args()
video_path = args.vid_path
out_path = args.out_path

assert out_path is not None, 'Give a valid output path so that I can write images to it.'

if not os.path.exists(out_path):
    os.makedirs(out_path)

cam_names = []
ipFinder = None
assert video_path is not None and video_path.isdigit(), 'make sure -vp is a digit representing the webcam'
video_path = int(video_path)
cam_name = 'Webcam{}'.format(video_path)

print('Video name: {}'.format(cam_name))
print('Video path: {}'.format(video_path))

vp = cv2.VideoCapture( video_path )
frame_count = 0

pose_idx = 0
# fill this up
poses = ['child', 'warrior', 'eagle', 'chair', 'leopard', 'salute', 'highkneel', 'handgun', 'crane', 'kunfusalute', 'chestbump', 'handshake', 'heart', 'dabbing', 'spiderman', 'hulk']

pic_idx = 0
for _, _, files in os.walk(out_path):
    pic_ids = [int(f[:-4]) for f in files if f.endswith('.png')]
    if len(pic_ids) > 0:
        pic_idx = max( pic_idx, max(pic_ids) )
pic_idx += 1
print( pic_idx )
#exit()

show_win_name = 'Capturing: {}'.format(poses[pose_idx])
cv2.namedWindow(show_win_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(show_win_name, 0, 0)

try:
    while True:
        status, frame = vp.read()
        # frame_show = cv2.resize(frame, (640*2,480*2))
        
        if status:
            cv2.imshow(show_win_name, frame )
            cv2.moveWindow(show_win_name, 0, 0)
            # cv2.imshow('Capturing: {}'.format(poses[pose_idx]),frame)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('c'):
            print('capture')
            cap_dir = os.path.join(out_path, poses[pose_idx])
            if not os.path.exists( cap_dir ):
                os.makedirs( cap_dir )
            cap_path = os.path.join(cap_dir, '{}.png'.format(pic_idx))
            cv2.imwrite( cap_path, frame )
            pic_idx += 1
        elif k == ord('n'):
            pose_idx += 1
            pose_idx = pose_idx % len( poses )
            #if pose_idx >= len( poses ):
            #    print('Exiting because all poses done.')
            #    break
            cv2.destroyAllWindows()
            show_win_name = 'Capturing: {}'.format(poses[pose_idx])
            cv2.namedWindow(show_win_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(show_win_name, 0, 0)
        elif k == ord('q'):
            break
        frame_count += 1

except KeyboardInterrupt:
    # print('Avg FPS:', frame_count/(time.time()-start_whole))
    print('KeyboardInterrupt:')
    print('Killing FrameGrabber..')
    os._exit(0)

cv2.destroyAllWindows()
print('Killing FrameGrabber..')
os._exit(0)
