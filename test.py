import os

import numpy as np

from pprint import pprint

from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import keras
import kerasapps.keras_applications
kerasapps.keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)

# from mobilenetv2 import relu6
# import tensorflow as tf


from kerasapps.keras_applications.resnet import preprocess_input as res_preproc
from keras.applications.resnet50 import preprocess_input as res50_preproc
from keras.applications.inception_v3 import preprocess_input as inv3_preproc
from keras.applications.inception_resnet_v2 import preprocess_input as inresv2_preproc
from keras.applications.xception import preprocess_input as xcep_preproc

from utils.preprocess_finder import preprocess_finder

finder = preprocess_finder(verbose=False)
# pp_dict = {'resnet152': res_preproc, 'resnet50':res50_preproc, 'xception':xcep_preproc, 'inception_v3':inv3_preproc, 'inception_resnet_v2':inresv2_preproc}
# pp_dict = {'resnet152': res_preproc, 'resnet50':res50_preproc, 'xception': lambda x: x/255., 'inception_v3':lambda x: x/255., 'inception_resnet_v2':lambda x: x/255.}

def num_jpgs_nested_in_folder( folder ):
    total_imgs = 0
    for _,_,imgs in os.walk( folder ):
        total_imgs += len([im for im in imgs if im.endswith(('.jpg','.png'))])
    return total_imgs

# def run_models_single_mode():
#     models_path = '/home/angeugn/Workspace/aicamp/models/test_model'
#     # test_folder = 'data/yoga/test'
#     test_folder = '/home/angeugn/Workspace/aicamp/data/TIL2019_v0.1_yoloed/test'

#     target_size = (224,224)
#     bs = 1
#     # bs = 4
#     total_testimgs = num_jpgs_nested_in_folder( test_folder )
#     steps = total_testimgs // bs
  
#     overall_results = {}
#     overall_results['all'] = {}
#     # model = load_model(model_path, custom_objects={'relu6':relu6, 'tf':tf})
#     for model_ in os.listdir(models_path):
#         if not model_.endswith('.hdf5'):
#             continue

#         context = model_[:model_.rfind('_')]

#         model_path = os.path.join( models_path, model_ )
#         print(model_path)
#         model = load_model(model_path)

#         test_datagen = ImageDataGenerator(preprocessing_function=finder(context))
#         test_generator = test_datagen.flow_from_directory(
#                 test_folder,
#                 target_size=target_size,
#                 batch_size=bs,
#                 class_mode='categorical')

#         results = model.evaluate_generator(test_generator, max_queue_size=bs, steps=steps, verbose=1)
#         for metric, scalar in zip( model.metrics_names, results ):
#             print('{}: {}'.format(metric, scalar))
#             overall_results['all'][model_] = (metric, scalar)
#             if metric not in overall_results:
#                 overall_results[metric] = scalar
#             elif overall_results[metric] < scalar:
#                 overall_results[metric] = scalar
#         del model
#     print('######### SUMMARY ##########')
#     pprint(overall_results)

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


def ensemble_models(models_path, test_folder, num_classes, save_preds=None):
    # models_path = '/home/angeugn/Workspace/aicamp/models/best_models'
    # test_folder = '/home/angeugn/Workspace/aicamp/data/TIL2019_v0.1_yoloed/test'
    class_labels = sorted( list(os.listdir(test_folder)) )

    target_size = (224,224)
    total_testimgs = num_jpgs_nested_in_folder( test_folder )

    all_preds = np.zeros( (total_testimgs, num_classes) )
    # model = load_model(model_path, custom_objects={'relu6':relu6, 'tf':tf})
    model_list = list(os.listdir(models_path))
    for model_ in model_list:
        if not model_.endswith('.hdf5'):
            continue
        model_path = os.path.join( models_path, model_ )
        model = load_model(model_path)
        context = model_[:model_.rfind('_')]
        print('Testing model: {}'.format(model_))

        i = 0

        for cl in class_labels:
            print('--> class: {}'.format(cl))
            pose_folder = os.path.join( test_folder, cl )
            sorted_imgs = sorted(list(os.listdir( pose_folder )))
            for im in sorted_imgs:
                img_path = os.path.join( pose_folder, im )
                img = image.load_img(img_path, target_size=target_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = finder(context)(x)
                preds = model.predict( x )
                all_preds[i] += preds[0]
                i+=1

        del model

    if save_preds is not None and type(save_preds) == str:
        np.save(save_preds, all_preds)

    eval_softmax_vectors(test_folder, all_preds)

def eval_models_singly(models_path, test_folder, num_classes):
    # models_path = '/home/angeugn/Workspace/aicamp/models/test_model'
    # test_folder = '/home/angeugn/Workspace/aicamp/data/TIL2019_v0.1_yoloed/test'
    class_labels = sorted( list(os.listdir(test_folder)) )

    target_size = (224,224)
    total_testimgs = num_jpgs_nested_in_folder( test_folder )

    # model = load_model(model_path, custom_objects={'relu6':relu6, 'tf':tf})
    model_list = list(os.listdir(models_path))
    for model_ in model_list:
        if not model_.endswith('.hdf5'):
            continue
        model_path = os.path.join( models_path, model_ )
        model = load_model(model_path)
        context = model_[:model_.rfind('_')]
        print('Testing model: {}'.format(model_))

        i = 0

        all_preds = np.zeros( (total_testimgs, num_classes) )
        for cl in class_labels:
            print('--> class: {}'.format(cl))
            pose_folder = os.path.join( test_folder, cl )
            sorted_imgs = sorted(list(os.listdir( pose_folder )))
            for im in sorted_imgs:
                img_path = os.path.join( pose_folder, im )
                img = image.load_img(img_path, target_size=target_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = finder(context)(x)
                preds = model.predict( x )
                all_preds[i] += preds[0]
                i+=1
        eval_softmax_vectors(test_folder, all_preds)

        del model



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    num_classes = 16
    eval_models_singly(models_path, test_folder, num_classes)
    # ensemble_models(models_path, test_folder, num_classes, save_preds='crop_preds.npy')
