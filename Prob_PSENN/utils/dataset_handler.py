import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from mnist import MNIST as mnist_loader
import sys
import scipy.io as sio
from skimage import transform


mnist_labels    = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fashion_labels  = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
hiragana_labels = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
emnist_bal_labels = ['0','1','2','3','4','5','6','7','8','9', 
                     'A','B','C','D','E','F','G','H','I','J','K','L','M',
                     'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                     'a','b', 'd','e','f','g','h', 'n', 'q','r', 't']
svhn_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_task_labels(problem):
    if    problem=="mnist":    out = mnist_labels
    elif  problem=="fashion":  out = fashion_labels
    elif  problem=="hiragana": out = hiragana_labels
    elif  problem=="emnist"  : out = emnist_bal_labels
    elif  problem=="svhn"    : out = svhn_labels
    else: sys.exit("Options: [mnist, fashion, hiragana, emnist, svhn]")
    return out


def get_task_name_capitalized(problem):
    if    problem=="mnist":    out = "MNIST"
    elif  problem=="fashion":  out = "F-MNIST"
    elif  problem=="hiragana": out = "K-MNIST"
    elif  problem=="emnist"  : out = "EMNIST"
    elif  problem=="svhn"    : out = "SVHN"
    else: sys.exit("Options: [mnist, fashion, hiragana, emnist, svhn]")
    return out

def get_and_adapt_labels(dataset_name, n_classes, filter_classes):
    labels = [i for i in range(n_classes)]
    labels_str = get_task_labels(dataset_name)

    #Preserve only the filtered classes, if any
    if not (filter_classes is None):
        labels = [i for i in filter_classes]
        labels_str = [labels_str[i] for i in filter_classes]

    return labels, labels_str


def get_load_configuration(dataset_name, ae_type):
    flatten    = True
    resize32   = False
    if dataset_name in ["svhn"]:
        flatten    = False
    if (ae_type in ["cnn_32"]) and (dataset_name in ["mnist", "fashion", "hiragana", "emnist"]):
        flatten  = False
        resize32 = True
    onehot     = True
    preprocess = True
    return flatten, resize32, onehot, preprocess


def load_dataset(dataset_name, batch_size, flatten, onehot, preprocess, filter_classes, split_config, resize32, valid_percentage,
                 shuffle=True, shuffle_epoch=True):
    if dataset_name == "mnist":
        #split_config = ['train[0%:%d%]'%train_percentage, 'train[%d%:]'%train_percentage, 'test']
        ds_list, input_shape, n_classes = get_mnist(batch_size=batch_size,
                                                    flatten=flatten,
                                                    onehot=onehot,
                                                    preprocess=preprocess,
                                                    filter_classes=filter_classes,
                                                    split_config=split_config,
                                                    resize32=resize32,
                                                    shuffle=shuffle,
                                                    shuffle_epoch=shuffle_epoch)

    if dataset_name == "fashion":
        ds_list, input_shape, n_classes = get_fashion_mnist(batch_size=batch_size,
                                                            flatten=flatten,
                                                            onehot=onehot,
                                                            preprocess=preprocess,
                                                            filter_classes=filter_classes,
                                                            valid_percentage=valid_percentage,
                                                            resize32=resize32,
                                                            shuffle=shuffle,
                                                            shuffle_epoch=shuffle_epoch)
    if dataset_name == "hiragana":
        tmp_data_path = 'datasets/hiragana/'
        ds_list, input_shape, n_classes = load_local_dataset(dir_path=tmp_data_path,
                                                        batch_size=batch_size,
                                                        flatten=flatten,
                                                        onehot=onehot,
                                                        preprocess=preprocess,
                                                        filter_classes=filter_classes,
                                                        valid_percentage=valid_percentage,
                                                        resize32=resize32,
                                                        shuffle=shuffle,
                                                        shuffle_epoch=shuffle_epoch)
        
    if dataset_name == "emnist":
        tmp_data_path = 'datasets/emnist_balanced/'
        ds_list, input_shape, n_classes = load_local_dataset(dir_path=tmp_data_path,
                                                        batch_size=batch_size,
                                                        flatten=flatten,
                                                        onehot=onehot,
                                                        preprocess=preprocess,
                                                        filter_classes=filter_classes,
                                                        valid_percentage=valid_percentage,
                                                        resize32=resize32,
                                                        dataset_name=dataset_name,
                                                        shuffle=shuffle,
                                                        shuffle_epoch=shuffle_epoch)
        
    if dataset_name == "svhn":
        tmp_data_path = 'datasets/svhn/'
        ds_list, input_shape, n_classes = load_local_dataset(dir_path=tmp_data_path,
                                                        batch_size=batch_size,
                                                        flatten=flatten,
                                                        onehot=onehot,
                                                        preprocess=preprocess,
                                                        filter_classes=filter_classes,
                                                        valid_percentage=valid_percentage,
                                                        dataset_name=dataset_name,
                                                        shuffle=shuffle,
                                                        shuffle_epoch=shuffle_epoch)
    
    return ds_list, input_shape, n_classes





# Load the dataset
def load_local_dataset(dir_path,
                 batch_size=128, 
                 flatten=False, 
                 onehot=False, 
                 preprocess=True, 
                 filter_classes=None, 
                 valid_percentage=10, 
                 resize32=False,
                 dataset_name='hiragana',
                 shuffle=True,
                 shuffle_epoch=True):

    #DATA LOADING
    if dataset_name in ["hiragana", "emnist"]:
        mndata = mnist_loader(dir_path)
        (x_train_, y_train_) = mndata.load_training()
        (x_test_,  y_test_)  = mndata.load_testing()
        x_train, y_train = np.array(x_train_).reshape(-1,28,28), np.array(y_train_)
        x_test,  y_test  = np.array(x_test_).reshape(-1,28,28),  np.array(y_test_)
        img_size_x, img_size_y, img_channels = 28, 28, 1
        input_shape = (img_size_x, img_size_y) # Input shapes
    elif dataset_name in ["svhn"]:
        #Load train and test shapes
        train_data = sio.loadmat(dir_path + "/train_32x32.mat")
        test_data  = sio.loadmat(dir_path + "/test_32x32.mat")
        x_train, y_train = train_data["X"], train_data["y"][:,0]
        x_test,  y_test  =  test_data["X"],  test_data["y"][:,0]
        #Shapes
        img_size_x, img_size_y, img_channels = 32, 32, 3
        input_shape = (img_size_x, img_size_y, img_channels) # Input shapes
                          
    #DATA FORMATTING
    if dataset_name=="emnist":
        #images are inverted horizontally and rotated 90 anti-clockwise, and thus should be transposed
        x_train = np.transpose(x_train, axes=[0,2,1])
        x_test  = np.transpose(x_test, axes=[0,2,1])
    elif dataset_name=="svhn":
        #set class "10" to class "0"
        y_train[y_train==10]=0
        y_test[y_test==10]=0
        #Reformat the axis
        x_train = x_train.transpose(3,0,1,2)
        x_test  = x_test.transpose(3,0,1,2)
                          

    #n_classes = 10  # Number of classes
    n_classes = len(set(y_train))

    if valid_percentage>0:
        valid_prop = valid_percentage / 100.0
        n_total_train  = x_train.shape[0]

        n_train = int(n_total_train * (1.0 - valid_prop))
        n_valid = int(n_total_train * valid_prop)
        #if dataset_name in ["hiragana", "emnist"]:
        if (n_train + n_valid) != n_total_train:
            print("WARNING! Total samples (%d) and split-sizes are train (%d) - valid (%d)"%(n_total_train,
                                                                                             n_train,
                                                                                             n_valid))

        if shuffle:
            rndstate = np.random.RandomState(1234)
            indices = rndstate.permutation(x_train.shape[0])
        else:
            indices = np.arange(x_train.shape[0])
        indices_train = np.copy(indices[:n_train])
        indices_valid = np.copy(indices[n_train:])

        x_valid, y_valid = np.copy(x_train[indices_valid]), np.copy(y_train[indices_valid])
        x_train, y_train = np.copy(x_train[indices_train]), np.copy(y_train[indices_train])

    #Create tensorflow Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    if valid_percentage>0:
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        ds_list = [train_ds, valid_ds, test_ds]
    else:
        ds_list = [train_ds, test_ds]

    # Preprocess train partition
    for i in range(len(ds_list)):
        ds_list[i] = ds_list[i].map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))  # casting inputs
        if not (filter_classes is None):
            ds_list[i] = ds_list[i].filter(lambda x,y: tf.reduce_any(tf.equal(y, filter_classes))) #filter classes
            n_classes  = len(filter_classes)
            ds_list[i] = ds_list[i].map(lambda x,y: (x, tf.where(tf.equal(y, filter_classes))[0][0])) #conver new classes into [0..n_classes]
        if preprocess:
            ds_list[i]  = ds_list[i].map(lambda x, y: (x * 2. - 1., y))  # convert to the range [-1., 1.]
        if flatten:
            input_shape = (img_size_x * img_size_y * img_channels,)  # Flattened shape
            ds_list[i]  = ds_list[i].map(lambda x, y: (tf.reshape(x, (img_size_x * img_size_y * img_channels,)), y)) #Flaten inputs
        if resize32:
            assert dataset_name in ["hiragana", "emnist"], "Resize32 only supported in hiragana and emnist"
            input_shape = (32, 32)  # Flattened shape
            ds_list[i]  = ds_list[i].map(lambda x, y: 
                            (tf.reshape(tf.image.resize(tf.reshape(x,[28,28,1]), [32,32]), [32,32]), y)
                                        ) #Flaten inputs
        if onehot:
            ds_list[i]  = ds_list[i].map(lambda x, y: (x, tf.one_hot(y, n_classes)))  #Convert labels from ints to one-hot-vecs
        ds_list[i] = ds_list[i].cache()
        cur_cardinality = len(list(ds_list[i]))
        if shuffle_epoch:
            ds_list[i] = ds_list[i].shuffle(cur_cardinality)
        ds_list[i] = ds_list[i].batch(batch_size)
        ds_list[i] = ds_list[i].prefetch(tf.data.experimental.AUTOTUNE)

    return ds_list, input_shape, n_classes




def get_fashion_mnist(batch_size=128, 
                      flatten=False, 
                      onehot=False, 
                      preprocess=True, 
                      filter_classes=None, 
                      valid_percentage=10, 
                      resize32=False,
                      dataset_name='fashion_mnist',
                      shuffle=True,
                      shuffle_epoch=True):
    
    #ds, ds_info = tfds.load(dataset_name)
    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Number of classes
    n_classes = 10
    # Input shapes
    input_shape = x_train.shape[1:]
    img_size_x, img_size_y = input_shape[0], input_shape[1]
    
    if valid_percentage>0:
        valid_prop = valid_percentage / 100.0
        n_total_train  = x_train.shape[0]

        n_train = int(n_total_train * (1.0 - valid_prop))
        n_valid = int(n_total_train * valid_prop)
        assert (n_train + n_valid) == n_total_train, "Check the dataset splitting"

        if shuffle:
            rndstate = np.random.RandomState(1234)
            indices  = rndstate.permutation(x_train.shape[0])
        else:
            indices  = np.arange(x_train.shape[0])

        indices_train = np.copy(indices[:n_train])
        indices_valid = np.copy(indices[n_train:])

        x_valid, y_valid = np.copy(x_train[indices_valid]), np.copy(y_train[indices_valid])
        x_train, y_train = np.copy(x_train[indices_train]), np.copy(y_train[indices_train])
        
    #Create tensorflow Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    if valid_percentage>0:
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        ds_list = [train_ds, valid_ds, test_ds]
    else:
        ds_list = [train_ds, test_ds]
        
    # Preprocess train partition
    for i in range(len(ds_list)):
        ds_list[i] = ds_list[i].map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))  # casting inputs
        if not (filter_classes is None):
            ds_list[i] = ds_list[i].filter(lambda x,y: tf.reduce_any(tf.equal(y, filter_classes))) #filter classes
            n_classes  = len(filter_classes)
            ds_list[i] = ds_list[i].map(lambda x,y: (x, tf.where(tf.equal(y, filter_classes))[0][0])) #conver new classes into [0..n_classes]
        if preprocess:
            ds_list[i]  = ds_list[i].map(lambda x, y: (x * 2. - 1., y))  # convert to the range [-1., 1.]
        if flatten:
            input_shape = (img_size_x * img_size_y,)  # Flattened shape
            ds_list[i]  = ds_list[i].map(lambda x, y: (tf.reshape(x, (img_size_x * img_size_y,)), y)) #Flaten inputs
        if resize32:
            input_shape = (32, 32)  # Flattened shape
            ds_list[i]  = ds_list[i].map(lambda x, y: 
                            (tf.reshape(tf.image.resize(tf.reshape(x,[28,28,1]), [32,32]), [32,32]), y)
                                        ) #Flaten inputs
        if onehot:
            ds_list[i]  = ds_list[i].map(lambda x, y: (x, tf.one_hot(y, n_classes)))  #Convert labels from ints to one-hot-vecs
        ds_list[i] = ds_list[i].cache()
        cur_cardinality = len(list(ds_list[i]))
        if shuffle_epoch:
            ds_list[i] = ds_list[i].shuffle(cur_cardinality)
        ds_list[i] = ds_list[i].batch(batch_size)
        ds_list[i] = ds_list[i].prefetch(tf.data.experimental.AUTOTUNE)

    return ds_list, input_shape, n_classes



def get_mnist(batch_size=128, flatten=False, onehot=False, preprocess=True, filter_classes=None, 
              split_config=['train', 'test'], resize32=False, dataset_name='mnist', 
              shuffle=True, shuffle_epoch=True):
    # Load the dataset
    ds_list, ds_info = tfds.load(dataset_name,
                                 split=split_config,
                                 shuffle_files=shuffle,
                                 as_supervised=True,
                                 with_info=True,
                                )
    print("Loaded Dataset:")
    #print(ds_info)

    # Number of classes
    n_classes = ds_info.features["label"].num_classes
    # Input shapes
    input_shape = ds_info.features["image"].shape
    img_size_x  = input_shape[0]
    img_size_y  = input_shape[1]
    #Split sizes
    #split_sizes = [len(ds_list[i]) for i in range(len(split_config))]

    # Preprocess train partition
    for i in range(len(split_config)):
        ds_list[i] = ds_list[i].map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))  # casting inputs
        if not (filter_classes is None):
            ds_list[i] = ds_list[i].filter(lambda x,y: tf.reduce_any(tf.equal(y, filter_classes))) #filter classes
            n_classes  = len(filter_classes)
            ds_list[i] = ds_list[i].map(lambda x,y: (x, tf.where(tf.equal(y, filter_classes))[0][0])) #conver new classes into [0..n_classes]
        if preprocess:
            ds_list[i]  = ds_list[i].map(lambda x, y: (x * 2. - 1., y))  # convert to the range [-1., 1.]
        if flatten:
            input_shape = (img_size_x * img_size_y,)  # Flattened shape
            ds_list[i]  = ds_list[i].map(lambda x, y: (tf.reshape(x, (img_size_x * img_size_y,)), y)) #Flaten inputs
        if resize32:
            input_shape = (32, 32)  # Flattened shape
            ds_list[i]  = ds_list[i].map(lambda x, y: 
                            (tf.reshape(tf.image.resize(tf.reshape(x,[28,28,1]), [32,32]), [32,32]), y)
                                        ) #Flaten inputs
        if onehot:
            ds_list[i]  = ds_list[i].map(lambda x, y: (x, tf.one_hot(y, n_classes)))  #Convert labels from ints to one-hot-vecs
        ds_list[i] = ds_list[i].cache()
        #ds_list[i] = ds_list[i].shuffle(ds_info.splits['train'].num_examples)
        cur_cardinality = len(list(ds_list[i]))
        if shuffle_epoch:
            ds_list[i] = ds_list[i].shuffle(cur_cardinality)
        ds_list[i] = ds_list[i].batch(batch_size)
        ds_list[i] = ds_list[i].prefetch(tf.data.experimental.AUTOTUNE)

    return ds_list, input_shape, n_classes



# Converts the data from [0,1] to [-1,1]
def normalize_data(data):
    return data*2.0 - 1.0

# Converts the data from [-1,1] to [0,1]
def unnormalize_data(data):
    return (data + 1.0) / 2.0


#Unpickle (useful to load certain datasets)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def resize_batch_to32(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1), dtype=np.float32)
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
        
    resized_imgs = resized_imgs.reshape(-1,32,32)
    return resized_imgs


def resize_batch_to28(imgs):
    # A function to resize a batch of MNIST images to (28, 28)
    # Args:
    #   imgs: a numpy array of size [batch_size, 32 X 32].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 32, 32, 1))
    resized_imgs = np.zeros((imgs.shape[0], 28, 28, 1), dtype=np.float32)
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (28, 28))
        
    resized_imgs = resized_imgs.reshape(-1,28,28)
    return resized_imgs

def get_shape_full_channels(dataset_name, input_shape):
    if len(input_shape)!=1:
        return input_shape 
    if dataset_name in ["mnist", "fashion", "hiragana", "emnist"]:
        return (28,28)
    else:
        sys.exit("Undefined scenario")
    





