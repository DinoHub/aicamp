from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from mobilenetv2 import relu6
import tensorflow as tf

if __name__ == '__main__':
	model_path = 'models/tbtest.82-2.39.hdf5'
	target_size = (224,224)
	bs = 30
	steps = 450 // bs

	test_folder = 'data/baseline/test'
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(
	        test_folder,
	        target_size=target_size,
	        batch_size=bs,
	        class_mode='categorical')
	
	model = load_model(model_path, custom_objects={'relu6':relu6, 'tf':tf})

	results = model.evaluate_generator(test_generator, max_queue_size=bs, steps=steps, verbose=1)
	for metric, scalar in zip( model.metrics_names, results ):
		print('{}: {}'.format(metric, scalar))
