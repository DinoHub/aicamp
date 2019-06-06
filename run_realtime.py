import cv2
import numpy as np
import time
import keras
import kerasapps.keras_applications
kerasapps.keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)

from keras.models import load_model
from kerasapps.keras_applications.mobilenet_v2 import preprocess_input
# from PIL import Image
from keras_yolo.yolo import YOLO

if __name__ == '__main__':
    od_threshold = 0.3
    od = YOLO(threshold=od_threshold)
    print('Human Detector loaded!')

    model_path = '/home/dh/Workspace/aicamp/models/mobilenet_v2/mobilenet_v2_acc.hdf5'
    # model_path = '/home/angeugn/Workspace/aicamp/models/mobilenet_v2/mobilenet_v2_acc.hdf5'
    model = load_model(model_path)
    model.summary()

    model_path1 = '/home/dh/Workspace/aicamp/models/mobilenet_v2/mobilenet_v2_1_acc.hdf5'
    # model_path1 = '/home/angeugn/Workspace/aicamp/models/mobilenet_v2_1/mobilenet_v2_1_acc.hdf5'
    model1 = load_model(model_path1)

    print('Classification Models loaded!')

    poses = ['ChairPose', 'ChestBump', 'ChildPose', 'Dabbing', 'EaglePose', 'HandGun', 'HandShake', 'HighKneel', 'HulkSmash', 'KoreanHeart', 'KungfuCrane', 'KungfuSalute', 'LeopardCrawl', 'Salute', 'Spiderman', 'WarriorPose']

    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0,255,255)
    fontScale = 2.0
    fontThickness = 2

    cap = cv2.VideoCapture(0)
    show_win_name = 'Pose for me!'
    cv2.namedWindow(show_win_name, cv2.WINDOW_NORMAL)

    while True:
        status, frame = cap.read()
        h,w,_ = frame.shape

        tic = time.time()
        chip, bb = od.get_largest_person_and_bb(frame, buf=0.1)
        toc = time.time()
        od_dur = toc - tic
        print('OD time:{}'.format(od_dur))

        chip_resized = cv2.resize( chip, (224,224) )

        x = np.expand_dims(chip_resized, axis=0)
        x = preprocess_input(x)

        tic = time.time()
        pred = model.predict( x )[0]
        toc = time.time()
        model0_dur = toc - tic
        print('Model0 time:{}'.format(model0_dur))

        tic = time.time()
        pred1 = model1.predict( x )[0]
        toc = time.time()
        model1_dur = toc - tic
        print('Model1 time:{}'.format(model1_dur))

        conf = np.max( pred + pred1 ) / 2.0 * 100.0
        class_ = poses[ np.argmax( pred + pred1 ) ]
        text = '{}:{:0.2f}%'.format(class_, conf)
        cv2.putText(frame, text, (10, 50), font, fontScale, color, fontThickness )
        t, l, b, r = bb
        cv2.rectangle(frame, (l,t), (r,b), color, 3)
        cv2.imshow( show_win_name, frame )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
