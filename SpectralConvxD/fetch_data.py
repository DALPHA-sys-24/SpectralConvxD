import tensorflow as tf
import numpy as np
import pandas as pd
from mnist1d.data import make_dataset, get_dataset_args,get_dataset


def generate_data(color_mode='grayscale',seed=23,subset="both",validation_split=0.2,name_data=None,file_path : str =None,n : int =None, p : int=None):
    if name_data=="mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        return x_train, y_train,x_test, y_test
    
    if name_data=="mnist1d":
        input_dim = 40
        defaults = get_dataset_args()
        defaults.num_samples = 10000
        #data = make_dataset(defaults)
        data=get_dataset(args=defaults, download=False, regenerate=True)
        x_train, y_train, x_test,y_test = data['x'], data['y'], data['x_test'],data['y_test']
        x_train, x_test = x_train.reshape(-1,input_dim), x_test.reshape(-1,input_dim)
        
        return x_train, y_train,x_test, y_test
    
    elif name_data=="fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        return x_train, y_train,x_test, y_test

    elif name_data == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train.reshape(-1,n, p, 3), x_test.reshape(-1, n, p, 3)
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return x_train, y_train, x_test, y_test
        
    elif name_data=="emnist":
        # Use pandas to read the CSV file into a DataFrame
        df = pd.read_csv(file_path,header=None)
        
        #split
        data_numpy=df.to_numpy()
        np.random.shuffle(data_numpy)
        train_size = int(0.8 * len(data_numpy))  # 80% for training
        test_size = len(data_numpy) - train_size
        x_train, x_test = data_numpy[:train_size, :-1], data_numpy[test_size:, :-1]  # Features
        y1, y2 = data_numpy[:train_size, -1], data_numpy[test_size:, -1]  #targets
        
        #transform
        letters={"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,
                "I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,
                "Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,
                "Y":24,"Z":25}
        y_train=np.array([letters[letter] for letter in y1])
        y_test=np.array([letters[letter] for letter in y2])
        
        #reshape
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train=np.reshape(x_train,newshape=(-1,n,p,1))
        x_test=np.reshape(x_test,newshape=(-1,n,p,1))
        
        #cast
        x_train=x_train.astype(np.float32)
        x_test=x_test.astype(np.float32)

            
        return x_train, y_train,x_test, y_test

    elif name_data=="face_mask":
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=file_path, # Path to the directory
        labels='inferred', # Automatically infers labels from directory structure
        label_mode='int', # Labels are integers
        color_mode=color_mode, # Images are color
        batch_size=853   , # Size of the batches of data
        image_size=(n, p), # Size to resize images to
        validation_split=validation_split,
        subset=subset,
        seed=seed,
        encodings='utf-8'
        )
        if subset=="both":
            [(x_train, y_train)] = dataset[0].take(1)
            [(x_test, y_test) ] = dataset[1].take(1)
        else :
            raise NotImplemented
            
        return x_train, y_train,x_test, y_test

    elif name_data=="pistachio":
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=file_path, # Path to the directory
        labels='inferred', # Automatically infers labels from directory structure
        label_mode='int', # Labels are integers
        color_mode=color_mode, # Images are color
        batch_size=2148   , # Size of the batches of data
        image_size=(n,p), # Size to resize images to
        validation_split=validation_split,
        subset=subset,
        seed=seed,
        )
        if subset=="both":
            [(x_train, y_train)] = dataset[0].take(1)
            [(x_test, y_test) ] = dataset[1].take(1)
        else :
            raise NotImplemented
            
        return x_train, y_train,x_test, y_test
    else:
         raise NotImplemented
#-----------------------------------------------------------------------#
#Pistachio_Image_Dataset/Pistachio_Image_Dataset,Done
#face_mask_data, Done
#fashion_mnist,Done
#emnist, Done
#mnist;Done
#-----------------------------------------------------------------------#
