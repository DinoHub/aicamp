import numpy as np
import os 
from PIL import Image
from skimage.transform import resize
import keras

class NpyDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_root, list_IDs, labels, batch_size=32, dim=(224,224), n_channels=81,
                 n_classes=15, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.data_root = data_root
        assert os.path.isdir(self.data_root),'data root given not a directory!'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = np.load(os.path.join(self.data_root,ID+'{}'.format('.npy' if not ID.endswith('.npy') else '')))
            # X[i,] = np.load(os.path.join(self.data_root,ID+'{}'.format('.npy' if not ID.endswith('.npy') else '')))
            # img = img.astype('uint8')[:,:,:5]
            # im = Image.fromarray(img)
            # im.thumbnail(self.dim[:2], Image.ANTIALIAS)
            resized = resize(img, self.dim[:2], anti_aliasing=True)
            X[i,] = resized/255.

            # Store class
            y[i] = self.labels[i]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# if __name__ == '__main__':
#     data_root = 
#     list_IDs = 
#     labels = 
#     gen = NpyDataGenerator(data_root, list_IDs, labels)