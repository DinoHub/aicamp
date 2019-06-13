#!/usr/bin/python3
from threading import Thread
import sys
import cv2
import os
from queue import Queue
from collections import deque
if os.path.basename(sys.path[0]) == 'utils':
    from misc import ping
else:
    from .misc import ping
import time
class VideoStream:
    def __init__(self, camName, vidPath, ipFinder=None, queueSize=5, writeDir='./outFrames/', reconnectThreshold=20):
        # self.stream = cv2.VideoCapture(vidPath, cv2.CAP_GSTREAMER)
        self.stream = cv2.VideoCapture(vidPath)
        self.camName = camName
        self.vidPath = vidPath
        # now, VideoStream is never stopped, just waits in a while True loop for the next ret frame.
        self.stopped = False 
        self.Q = deque(maxlen=queueSize)
        assert self.stream.isOpened(), 'error opening video file'
        print('VideoStream for {} initialised!'.format(self.camName))
        self.writeDir = writeDir
        if not os.path.isdir(writeDir):
            os.mkdir(writeDir)
        self.writeCount = 1
        self.reconnectThreshold = reconnectThreshold
        self.pauseTime = None
        self.ipFinder = ipFinder

    def getInfo(self):
        video_info = {}
        video_info['width'] = int(self.stream.get(3))
        video_info['height'] = int(self.stream.get(4))
        video_info['fps'] = self.stream.get(cv2.CAP_PROP_FPS)
        # video_info['start_time'] = 0 #in secs elapsed
        # video_info['context'] = context
        # video_info['cam'] = cam
        return video_info

    def start(self):
        t = Thread(target=self._update, args=())
        t.daemon = True
        t.start()
        print('VideoStream started')
        # return self

    def reconnect_start(self):
        s = Thread(target=self.reconnect, args=())
        s.daemon = True
        s.start()

    def _update(self):
        while True:
            # if self.stopped:
                # return
            assert self.stream.isOpened(),'OHNO STREAM IS CLOSED.'
            try:
                # print(self.camName,'trying to grab')
                ret, frame = self.stream.read()
                if ret: 
                    self.Q.appendleft(frame)
                    # print('Grabbed')
            except Exception as e:
                print('stream.grab error:{}'.format(e))
                ret = False
            if not ret:
                # print(self.camName,'no Ret!')
                if self.pauseTime is None:
                    self.pauseTime = time.time()
                    self.printTime = time.time()
                    print('No frames for {}, starting {:0.1f}sec countdown to reconnect.'.\
                            format(self.camName,self.reconnectThreshold))
                time_since_pause = time.time() - self.pauseTime
                time_since_print = time.time() - self.printTime
                if time_since_print > 5: #prints only every 5 sec
                    print('No frames for {}, reconnect starting in {:0.1f}sec'.\
                            format(self.camName,self.reconnectThreshold-time_since_pause))
                    self.printTime = time.time()
                        
                if time_since_pause > self.reconnectThreshold:
                    self.reconnect_start()
                    break
                continue
                # self.stop()
                # return
            # if not self.Q.full():
            self.pauseTime = None
            # if ret:
                # ret, frame = self.stream.retrieve()
                # print(self.camName,'ret for retrieve:',ret)
                # print(frame.shape)

    def _updateOld(self):
        while True:
            # if self.stopped:
                # return
            assert self.stream.isOpened(),'OHNO STREAM IS CLOSED.'
            try:
                # print(self.camName,'trying to grab')
                ret = self.stream.grab()
                # print('Grabbed')
                #TODO: stream grabbing is getting stuck ocassionally
            except Exception as e:
                print('stream.grab error:{}'.format(e))
                ret = False
            if not ret:
                # print(self.camName,'no Ret!')
                if self.pauseTime is None:
                    self.pauseTime = time.time()
                    self.printTime = time.time()
                    print('No frames for {}, starting {:0.1f}sec countdown to reconnect.'.\
                            format(self.camName,self.reconnectThreshold))
                time_since_pause = time.time() - self.pauseTime
                time_since_print = time.time() - self.printTime
                if time_since_print > 5: #prints only every 5 sec
                    print('No frames for {}, reconnect starting in {:0.1f}sec'.\
                            format(self.camName,self.reconnectThreshold-time_since_pause))
                    self.printTime = time.time()
                        
                if time_since_pause > self.reconnectThreshold:
                    self.reconnect_start()
                    break
                continue
                # self.stop()
                # return
            # if not self.Q.full():
            self.pauseTime = None
            # print(self.camName,'pauseTime set to None')
            # if ret and self.Q.full():
            #     print(self.camName,'there is ret, but Q is full')
            # if ret and not self.Q.full():
            if ret:
                ret, frame = self.stream.retrieve()
                # print(self.camName,'ret for retrieve:',ret)
                self.Q.appendleft(frame)
                # print(frame.shape)

    def read(self):
        self.currentFrame = self.Q.pop()
        return self.currentFrame

    def more(self):
        # return self.Q.qsize() > 0
        return bool(self.Q)

    def stop(self):
        self.stopped = True
        self.stream.release()

    def capture(self):
        path = os.path.join(self.writeDir, '{}_{}.png'.format(self.camName, str(self.writeCount)))
        cv2.imwrite(path, self.currentFrame)
        self.writeCount+=1

    def reconnect(self):
        print('Reconnecting to',self.camName)
        self.stream.release()
        # with self.Q.mutex:
        #     self.Q.queue.clear()
        self.Q.clear()
        while not self.stream.isOpened():
            if self.ipFinder is not None:
                self.vidPath = self.ipFinder.findStream(self.camName)
            self.stream = cv2.VideoCapture(self.vidPath)
        assert self.stream.isOpened(), 'error opening video file'
        print('VideoStream for {} initialised!'.format(self.vidPath))
        self.pauseTime = None
        self.start()