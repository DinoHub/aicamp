import os

import numpy as np

from pprint import pprint
from PIL import Image
from kerasyolo.yolo import YOLO

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import keras
import kerasapps.keras_applications
kerasapps.keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)

from kerasapps.keras_applications.resnet import preprocess_input as res_preproc
from keras.applications.resnet50 import preprocess_input as res50_preproc
from keras.applications.inception_v3 import preprocess_input as inv3_preproc
from keras.applications.inception_resnet_v2 import preprocess_input as inresv2_preproc
from keras.applications.xception import preprocess_input as xcep_preproc

from utils.preprocess_finder import finder

# finder = preprocess_finder(verbose=False)
od_threshold = 0.3
od = YOLO(threshold=od_threshold)

def num_jpgs_nested_in_folder( folder ):
    total_imgs = 0
    for _,_,imgs in os.walk( folder ):
        total_imgs += len([im for im in imgs if im.endswith(('.jpg','.png'))])
    return total_imgs

def eval_softmax_vectors(test_folder, preds):
    class_labels = sorted( list(os.listdir(test_folder)) )
    i=0
    score = 0
    for label, cl in enumerate(class_labels):
        pose_folder = os.path.join( test_folder, cl )
        for _ in os.listdir( pose_folder ):
            pred = np.argmax( preds[i] )
            if pred == label:
                score+=1
            i+=1

    print('Total evaluated: {}'.format(i))
    print('score: {}'.format(score))
    print('accuracy: {0:.6f}'.format( score / i ))


def ensemble_models(models_path, test_folder, num_classes, use_preds=False):
    total_testimgs = num_jpgs_nested_in_folder( test_folder )
    all_preds = np.zeros( (total_testimgs, num_classes) )
    if use_preds:
        for pred_file in [npy for npy in os.listdir('predictions') if npy.endswith('.npy')]:
            pred_fp = os.path.join('predictions', pred_file)
            this_preds = np.load(pred_fp)
            assert all_preds.shape == this_preds.shape, 'loaded shape of preds does not match the overall required shape'
            all_preds += this_preds
    else:
        # finder = preprocess_finder(verbose=False)
        od_threshold = 0.3
        od = YOLO(threshold=od_threshold)

        class_labels = sorted( list(os.listdir(test_folder)) )

        target_size = (224,224)

        model_list = list(os.listdir(models_path))
        for model_ in model_list:
            if not model_.endswith('.hdf5'):
                continue
            model_path = os.path.join( models_path, model_ )
            model = load_model(model_path)
            context = model_[:model_.rfind('_')]
            print('Running model {} with context {}'.format(model_, context))

            i = 0

            for cl in class_labels:
                print('--> class: {}'.format(cl))
                pose_folder = os.path.join( test_folder, cl )
                sorted_imgs = sorted(list(os.listdir( pose_folder )))
                for im in sorted_imgs:
                    img_path = os.path.join( pose_folder, im )
                    # img = image.load_img(img_path, target_size=target_size)
                    img = Image.open( img_path )
                    img = od.crop_largest_person( img ) 
                    img = img.resize( target_size, Image.NEAREST )
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = finder(context, verbose=False)(x)
                    preds = model.predict( x )
                    all_preds[i] += preds[0]
                    i+=1

            del model

    print('evaluating the ensemble.')

    eval_softmax_vectors(test_folder, all_preds)

def eval_models_singly(models_path, test_folder, num_classes, save_preds=False):
    finder = preprocess_finder(verbose=False)
    od_threshold = 0.3
    od = YOLO(threshold=od_threshold)

    class_labels = sorted( list(os.listdir(test_folder)) )

    target_size = (224,224)
    total_testimgs = num_jpgs_nested_in_folder( test_folder )

    model_list = list(os.listdir(models_path))
    for model_ in model_list:
        if not model_.endswith('.hdf5'):
            continue
        model_path = os.path.join( models_path, model_ )
        model = load_model(model_path)
        context = model_[:model_.rfind('_')]
        print('Testing model {} with context {}'.format(model_, context))

        i = 0

        all_preds = np.zeros( (total_testimgs, num_classes) )
        for cl in class_labels:
            print('--> class: {}'.format(cl))
            pose_folder = os.path.join( test_folder, cl )
            sorted_imgs = sorted(list(os.listdir( pose_folder )))
            for im in sorted_imgs:
                img_path = os.path.join( pose_folder, im )
                img = Image.open( img_path )
                img = od.crop_largest_person( img )
                img = img.resize( target_size, Image.NEAREST )

                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = finder(context)(x)
                preds = model.predict( x )
                all_preds[i] += preds[0]
                i+=1
        if save_preds:
            if not os.path.exists( 'predictions' ):
                os.makedirs('predictions')
            np.save('predictions/{}.npy'.format(context), all_preds)
        eval_softmax_vectors(test_folder, all_preds)
        del model



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('usage: python3 test_yolocrops.py path/to/models/folder path/to/test/folder')
        exit()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    models_path = sys.argv[1]
    test_folder = sys.argv[2]
    num_classes = 16
    # models_path = 'models/best_models'
    # test_folder = 'data/TIL2019_v0.1/test'
    # eval_models_singly(models_path, test_folder, num_classes, save_preds=True)
    ensemble_models(models_path, test_folder, num_classes, use_preds=True)
