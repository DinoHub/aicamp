import numpy as np
import cv2
import tkinter as tk
from threading import Thread
from time import sleep
from copy import deepcopy

class Name_chooser(tk.Frame):
    def __init__(self, parent, names, vidpath, out_path):
        self.parent = parent
        # self.frame = tk.Frame(self, parent, bg='#81ecec')
        tk.Frame.__init__(self, parent, bg='#81ecec')
        self.bind("<Key>", self.keydown)
        self.chosen = None
        # create a prompt, an input box, an output label,
        # and a button to do the computation
        self.prompt = tk.Label(self, text="Choose the name:",
                               anchor="w",
                               font=('Ubuntu Mono',15), 
                               fg='#000000',
                               bg='#81ecec',)
        self.names = names
        self.name_buttons = []
        for i, name in enumerate(names):
            self.name_buttons.append(
                tk.Button(self, text=name, 
                          font=('Ubuntu Mono',20), 
                          bg='#00b894',
                          activebackground='#0dd8b0',
                          width=20,
                          command = lambda i=i: self.choose(i)))
        self.custom = tk.Entry(self, font=('Ubuntu Mono',20), 
                               justify='center')
        self.submit_custom = tk.Button(self, 
                    text='Submit custom name', 
                    font=('Ubuntu Mono',20), 
                    bg='#a29bfe',
                    activebackground='#0dd8b0',
                    width=20,
                    command = self.custom_choose)
        # self.output = tk.Label(self, text="")

        global_pady = 4
        global_padx = 20
        # lay the widgets out on the screen. 
        self.prompt.pack(side="top", fill="x", pady=(5,global_pady), padx=global_padx)
        # self.output.pack(side="top", fill="x", expand=True)
        for button in self.name_buttons:
            button.pack(side="top", fill="x", pady=global_pady, padx=global_padx)
        self.custom.pack(side="top", fill="x", pady=(global_pady,0), padx=global_padx)
        self.submit_custom.pack(side="top", fill="x", pady=(0,15), padx=global_padx)
        self.chosen_name = None
        self.cv2_win_name = 'POSE'
        self.out_path = out_path
        self.video_cap = cv2.VideoCapture(vidpath)
        self.video_width = int(self.video_cap.get(3)) * 2
        self.video_height = int(self.video_cap.get(4)) * 2
        self.webcam_thread = None
        self.continue_thread = True
        self.show_frame = np.zeros((self.video_height, self.video_width))
        self.raw_frame = None
        self.pic_idx = 0
        self.pose_idx = -1
        self.modifier = ''
        self.webcam_thread = Thread(target=self.start_webcam, args=())
        self.webcam_thread.start()

        self.parent.protocol("WM_DELETE_WINDOW",self.on_exit)

    def capture(self):
        if self.chosen_name:
            t = Thread(target=self.SNAP,args=())
            t.start()
            print('capturing')
            cap_dir = os.path.join(self.out_path, self.chosen_name)
            if not os.path.exists( cap_dir ):
                os.makedirs( cap_dir )
            cap_path = os.path.join(cap_dir, '{}.png'.format(self.pic_idx))
            cv2.imwrite( cap_path, self.raw_frame )
            self.pic_idx += 1

    def next_pose(self):
        print('Moving on to next pose')
        self.pose_idx += 1
        self.pose_idx = self.pose_idx % len(self.names)
        self.chosen_name = self.names[self.pose_idx]
        update_thread = Thread(target=self.update_frame, args=(self.chosen_name,))
        update_thread.start()

    def keydown(self,e):
        # print('down |{}|'.format(e.char))
        if e.char == ' ' or e.char == 'c' or e.char=='C':
            self.capture()
        elif e.char =='n' or e.char=='N':
            self.next_pose()

    def update_frame(self, pose):
        while self.chosen_name == pose:
            ret, frame = self.video_cap.read()
            if ret:
                self.raw_frame = frame
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                text = pose
                cv2.putText(frame, text, 
                        (20, int(self.video_height/2)+50),  
                        cv2.FONT_HERSHEY_DUPLEX,
                        5, (255,255,0), 3)
                if self.modifier:
                    cv2.putText(frame, self.modifier, 
                            (0, int(self.video_height)-30),  
                            cv2.FONT_HERSHEY_DUPLEX,
                            3, (0,0,255), 5)
                    cv2.rectangle(frame, (0, 0), (self.video_width-1, self.video_height-1), (0,0,255), 30)
                self.show_frame = deepcopy(frame)

    def SNAP(self):
        self.modifier = "SNAPPED"
        sleep(0.3)
        self.modifier = ''

    def start_webcam(self):
        cv2.namedWindow(self.cv2_win_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.cv2_win_name, 400, 0)
        cv2.resizeWindow(self.cv2_win_name, (self.video_width, self.video_height))
        while self.show_frame is not None:
            cv2.imshow(self.cv2_win_name, self.show_frame)
            k = cv2.waitKey(5) & 0xFF
            if k == ord('q'):
                print('{} end'.format(self.chosen_name))
                self.chosen_name = None
                self.show_frame = np.zeros((self.video_height, self.video_width))
            elif self.chosen_name and (k == 32 or k == ord('c')):
                self.capture()
            elif k == ord('n'):
                self.next_pose()

    def choose(self, idx):
        self.pose_idx = idx
        self.chosen_name = self.names[idx]
        print('Chosen name:{}'.format(self.chosen_name))
        self.show_frame = np.ones((self.video_height, self.video_width)) * 255
        update_thread = Thread(target=self.update_frame, args=(self.chosen_name,))
        update_thread.start()
        # cv2.destroyAllWindows()
        # self.start_webcam(self.chosen_name)
        # self.webcam_thread = Thread(target=self.start_webcam, args=(self.chosen_name,))
        # self.webcam_thread.start()
        # self.parent.destroy()

    def custom_choose(self):
        self.chosen_name = self.custom.get()
        print('Chosen custom name:{}'.format(self.chosen_name))
        # self.parent.destroy()

    def on_exit(self):
        self.chosen_name = None
        self.show_frame = None
        print('Quiting')
        self.parent.destroy()


# if this is run as a program (versus being imported),
# create a root window and an instance of our example,
# then start the event loop

def poser(classes=['Tom', 'Dick', 'Harry'], frame_size=None,screen_loc=[0,0],  screen_size=None, bb_loc=None, video_path=None, out_path=None):
    
    root = tk.Tk()
    root.configure(bg='#81ecec')
    ws = root.winfo_screenwidth()  # width of screen
    hs = root.winfo_screenheight() # height of screen
    est_width = 400
    print(ws, hs)
    # frame_h, frame_w = frame_size
    # screen_w, screen_h = screen_size
    if bb_loc:
        ## TODO: this is iffy, bb_loc does not take into account scaling done if max window is hit.
        x = screen_loc[0] + bb_loc['rect']['r']
        if (x+est_width) > ws:
            x = screen_loc[0] + bb_loc['rect']['l'] - est_width

        # y = screen_loc[1] + bb_loc['rect']['t']
        y = screen_loc[1] + bb_loc['rect']['t']
    else:
        x = screen_loc[0]
        y = screen_loc[1]

    x = screen_loc[0]
    y = screen_loc[1]

    root.geometry('+%d+%d'%(int(x),int(y)))
    choser = Name_chooser(root, classes, video_path, out_path)
    choser.pack(fill="both", expand=True)
    choser.focus_set()
    root.mainloop()
    # print('out of loop Chosen:{}'.format(choser.names[choser.chosen]))
    chosen_name = choser.chosen_name
    if chosen_name is not None and chosen_name not in classes:
            classes.append(chosen_name)
    return chosen_name, classes

if __name__ == '__main__':
    import argparse 
    import os

    poses = ['KoreanHeart', 'Dabbing','Salute', 'KungfuSalute','HandGun','Spiderman','KungfuCrane','WarriorPose','EaglePose','ChairPose','HulkSmash','HighKneel','ChildPose','LeopardCrawl','Handshake','ChestBump']

    parser = argparse.ArgumentParser()
    parser.add_argument('-vp','--vid_path', help='Video filepaths/streams for \
                        all cameras, e.g.: 0')
    parser.add_argument('-out','--out_path', help='Output path to write images to')
    args = parser.parse_args()
    video_path = int(args.vid_path)
    out_path = args.out_path

    assert out_path is not None, 'Give a valid output path so that I can write images to it.'
    assert video_path >= 0, 'Give a valid webcam number.'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    poser(classes=poses, video_path=video_path, out_path=out_path)