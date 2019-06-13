import os
from sys import getsizeof
import cv2
import time

def ping(ip, count=3):
    response = os.system("ping -c {} {}".format(count, ip))
    # and then check the response...
    if response == 0:
        return True
    else:
    	return False

def repeat_ping(ip, count=3):
    retry = 1
    while True:
        connected = ping(ip, count=count)
        if connected:
            break
        else:
            print('{} not connected, retrying {}..'.format(ip, retry))
            time.sleep(5)
            retry += 1
    return True

# def justNiceShrink(img, mem_size=2e5):
def justNiceShrink(img, mem_size=5e5):
    size = getsizeof(img)
    while getsizeof(img) > mem_size:
        print('size of img too big:', getsizeof(img))
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        # cv2.imshow('',img)
        # cv2.waitKey(0) 
        # cv2.waitKey(0) & 0xFF
    print('resized of img:', size)
    return img


if __name__ == '__main__':
	rtsp = "rtsp://192.168.8.197/vga1"
	ip = rtsp.split('/')[2]
	print (ip)
	ret = ping(ip)
	print(ret)