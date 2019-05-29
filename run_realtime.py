import cv2

import numpy as np

import keras
import kerasapps.keras_applications
kerasapps.keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)

from keras.models import load_model
from kerasapps.keras_applications.mobilenet_v2 import preprocess_input
# from PIL import Image
# from kerasyolo.yolo import YOLO

if __name__ == '__main__':
	# od_threshold = 0.3
	# od = YOLO(threshold=od_threshold)

    model_path = '/home/angeugn/Workspace/aicamp/models/mobilenet_v2/mobilenet_v2_acc.hdf5'
    model = load_model( model_path )
    # model.summary()
    model_path1 = '/home/angeugn/Workspace/aicamp/models/mobilenet_v2_1/mobilenet_v2_1_acc.hdf5'
    model1 = load_model( model_path1 )

    poses = ['ChairPose', 'ChestBump', 'ChildPose', 'Dabbing', 'EaglePose', 'HandGun', 'HandShake', 'HighKneel', 'HulkSmash', 'KoreanHeart', 'KungfuCrane', 'KungfuSalute', 'LeopardCrawl', 'Salute', 'Spiderman', 'WarriorPose']

    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0,255,255)
    fontScale = 2.0
    fontThickness = 2

    cap = cv2.VideoCapture(0)

    while True:
        status, frame = cap.read()
        h,w,_ = frame.shape

        frame_resized = cv2.resize( frame, (224,224) )

        x = np.expand_dims(frame_resized, axis=0)
        x = preprocess_input(x)

        pred = model.predict( x )[0]
        pred1 = model1.predict( x )[0]
        text = poses[ np.argmax( pred + pred1 ) ]
        cv2.putText(frame, text, (int(w / 2), int(h / 2)), font, fontScale, color, fontThickness )
        cv2.imshow( '', frame )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()