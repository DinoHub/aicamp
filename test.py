import os

import numpy as np

from pprint import pprint

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

def num_jpgs_nested_in_folder( folder ):
    total_imgs = 0
    for _,_,imgs in os.walk( folder ):
        total_imgs += len([im for im in imgs if im.endswith(('.jpg','.png'))])
    return total_imgs

def eval_softmax_vectors(test_folder, preds):
    class_labels = sorted( list(os.listdir(test_folder)) )
    i=0
    score = 0
    filenames = []
    for label, cl in enumerate(class_labels):
        pose_folder = os.path.join( test_folder, cl )
        for fn in os.listdir( pose_folder ):
            filenames.append(fn)
            pred = np.argmax( preds[i] )
            if pred == label:
                score+=1
            i+=1

    print('Total evaluated: {}'.format(i))
    print('score: {}'.format(score))
    print('accuracy: {0:.6f}'.format( score / i ))
    return filenames

def ensemble_models(models_path, test_folder, num_classes, weights=None, use_preds=False):
    total_testimgs = num_jpgs_nested_in_folder( test_folder )
    all_preds = np.zeros( (total_testimgs, num_classes) )
    if use_preds:
        default_pred_dir = 'predictions_crops_test_tta5'
        for pred_file in [npy for npy in os.listdir(default_pred_dir) if npy.endswith('.npy')]:
            pred_fp = os.path.join(default_pred_dir, pred_file)

            weight = 1.0
            if weights is not None:
                print('using weight of {} = {}'.format(pred_file, weights[pred_file]))
                weight = weights[pred_file]

            print('Using {}'.format(pred_fp))
            this_preds = np.load(pred_fp)
            assert all_preds.shape == this_preds.shape, 'loaded shape of preds does not match the overall required shape'
            all_preds += this_preds * weight
    else:
        class_labels = sorted( list(os.listdir(test_folder)) )

        target_size = (224,224)
        # total_testimgs = num_jpgs_nested_in_folder( test_folder )

        # all_preds = np.zeros( (total_testimgs, num_classes) )
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
                    x = finder(context, verbose=False)(x)
                    preds = model.predict( x )
                    all_preds[i] += preds[0]
                    i+=1

            del model
    # all_preds /= total_models
    sums = np.sum( all_preds, axis=1 )
    for i in range(all_preds.shape[0]):
        all_preds[i] /= sums[i]
    filenames = eval_softmax_vectors(test_folder, all_preds)
    # np.save( 'predictions/fullcropped6_eugeneivanensemble.npy', all_preds )
    import pickle
    with open('fullcropped6_eugeneivanensemble_filenames.p', 'wb') as fnp:
        pickle.dump( filenames, fnp )
    np.savetxt('fullcropped6_eugeneivanensemble_preds.txt', all_preds)


alpha=0.5
num_classes=6
import keras.losses
import keras.backend as K
def custom_loss(ytrue, ypred):
    return alpha*K.categorical_crossentropy(ytrue[:,:num_classes], ypred) + (1. - alpha)*K.categorical_crossentropy(ytrue[:,num_classes:], ypred)
keras.losses.custom_loss = custom_loss

def tta(image, pp_fn, model):
    datagen = ImageDataGenerator(
            preprocessing_function=pp_fn, 
            # rotation_range=45,#in deg
            brightness_range= [0.5,1.5],
            # shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=True)
    samples = np.expand_dims(image, 0)
    it = datagen.flow(samples, batch_size=5)
    yhats = model.predict_generator(it, steps=5, verbose=0)
    summed = np.sum(yhats, axis=0)
    summed /= sum(summed)
    return [summed]

def eval_models_singly(models_path, test_folder, num_classes, save_preds=True, use_tta=True, preds_dir='predictions'):
    class_labels = sorted( list(os.listdir(test_folder)) )

    target_size = (224,224)
    total_testimgs = num_jpgs_nested_in_folder( test_folder )

    # model = load_model(model_path, custom_objects={'relu6':relu6, 'tf':tf})
    model_list = list(os.listdir(models_path))
    if save_preds:
        if not os.path.isdir(preds_dir):
            os.makedirs(preds_dir)
    for model_ in model_list:
        if not model_.endswith('.hdf5'):
            continue
        model_path = os.path.join( models_path, model_ )
        model = load_model(model_path)
        model.summary()
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
                if use_tta:
                    preds = tta( x, finder(context, verbose=False), model )
                else:
                    x = np.expand_dims(x, axis=0)
                    x = finder(context, verbose=False)(x)
                    preds = model.predict( x )
                all_preds[i] += preds[0]
                i+=1
        eval_softmax_vectors(test_folder, all_preds)
        if save_preds:
            np.save( os.path.join(preds_dir,'{}.npy'.format(context)), all_preds )
        del model

def conf_mat():
    from sklearn.metrics import classification_report, confusion_matrix
    cm = confusion_matrix()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('usage: python3 test.py path/to/models/folder path/to/test/folder')
        exit()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    models_path = sys.argv[1]
    test_folder = sys.argv[2]
    num_classes = 15
    # models_path = 'models/best_models'
    # test_folder = 'data/TIL2019_v0.1/test'
    eval_models_singly(models_path, test_folder, num_classes)
    # ensemble_models(models_path, test_folder, num_classes, use_preds=True)
