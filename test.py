import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# from mobilenetv2 import relu6
# import tensorflow as tf

def num_jpgs_nested_in_folder( folder ):
    total_imgs = 0
    for _,_,imgs in os.walk( folder ):
        total_imgs += len([im for im in imgs if im.endswith(('.jpg','.png'))])
    return total_imgs

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # model_path = '/home/dh/Workspace/aicamp/models/inception_resnet_v2_acc.102-0.922.hdf5'
    # models_path = '/home/dh/Workspace/aicamp/models/inception_resnet_v2_fulldata'
    models_path = '/home/dh/Workspace/aicamp/models/inception_resnet_v2_small_competition/'
    # test_folder = 'data/yoga/test'
    test_folder = 'data/competition_split/test'

    target_size = (224,224)
    bs = 64
    # bs = 4
    total_testimgs = num_jpgs_nested_in_folder( test_folder )
    steps = total_testimgs // bs
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            test_folder,
            target_size=target_size,
            batch_size=bs,
            class_mode='categorical')
  
    overall_results = {}
    overall_results['all'] = {}
    # model = load_model(model_path, custom_objects={'relu6':relu6, 'tf':tf})
    for model_ in os.listdir(models_path):
        if not model_.endswith('.hdf5'):
            continue
        model_path = os.path.join( models_path, model_ )
        print(model_path)
        model = load_model(model_path)
        results = model.evaluate_generator(test_generator, max_queue_size=bs, steps=steps, verbose=1)
        for metric, scalar in zip( model.metrics_names, results ):
            print('{}: {}'.format(metric, scalar))
            overall_results['all'][model_] = (metric, scalar)
            if metric not in overall_results:
                overall_results[metric] = scalar
            elif overall_results[metric] < scalar:
                overall_results[metric] = scalar
        del model
    print('######### SUMMARY ##########')
    print(overall_results)
