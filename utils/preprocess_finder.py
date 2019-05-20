import keras
import kerasapps

def preprocess_finder(verbose=True):
	def finder(context, verbose=verbose):
		if context == 'xception':
			if verbose:
				print('----> USING {} native preprocessing'.format(context))
			return keras.applications.xception.preprocess_input
		elif context == 'resnet50':
			if verbose:
				print('----> USING {} native preprocessing'.format(context))
			return keras.applications.resnet50.preprocess_input
		elif context == 'inception_resnet_v2':
			if verbose:
				print('----> USING {} native preprocessing'.format(context))
			return keras.applications.inception_resnet_v2.preprocess_input
		elif context == 'inception_v3':
			if verbose:
				print('----> USING {} native preprocessing'.format(context))
			return keras.applications.inception_v3.preprocess_input
		elif context == 'mobilenet_v2':
			if verbose:
				print('----> USING {} native preprocessing'.format(context))
			return keras.applications.mobilenet_v2.preprocess_input
		elif context.startswith('resnet') and context.endswith('_v2'):
			if verbose:
				print('----> USING {} native preprocessing'.format(context))
			return kerasapps.keras_applications.resnet_v2.preprocess_input
		elif context.startswith('resnet'):
			if verbose:
				print('----> USING {} native preprocessing'.format(context))
			return kerasapps.keras_applications.resnet.preprocess_input
		else:
			if verbose:
				print('----> USING 1 / 255. as preprocessing')
			return lambda x : x / 255.
	return finder