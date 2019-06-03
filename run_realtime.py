import cv2

from keras.models import load_model

if __name__ == '__main__':
	model_path = '/home/dh/Workspace/aicamp/models/mobilenet_v2/mobilenet_v2_acc.hdf5'

	model = load_model( model_path )


	font = cv2.FONT_HERSHEY_DUPLEX
	color = (0,255,255)
	fontScale = 2.0
	fontThickness = 2

	cap = cv2.VideoCapture(0)

	while True:
		status, frame = cap.read()
		h,w,_ = frame.shape

		pred = model.predict( frame )
		print(pred)
		exit()
		text = 'Hello'
		cv2.putText(frame, text, (int(w / 2), int(h / 2)), font, fontScale, color, fontThickness )
		cv2.imshow( '', frame )

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()