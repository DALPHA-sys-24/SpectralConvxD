import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
from mnist1d.data import make_dataset, get_dataset_args,get_dataset

from utils import*
from fetch_data import*
from specCnn2D import *
from specCnn1D import *
from SpectralLayer import*

def build_best_model(hp):

    d,c=40,1
    use_bias=True
    activation="relu"


    spectral_config = { 'is_base_trainable': True,
                    'is_diag_start_trainable': False,
                    'is_diag_end_trainable': True,
                    'use_bias': True
                    }

    model= tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(d,)))
    model.add(SpecCnn1D(filters=20,kernel_size=3,stride=hp.Choice('stride', values=[1,2,3]),padding=0,trainable_phi=True,use_lambda_out=False,use_lambda_in=False,activation=activation,use_bias=use_bias))
    
    model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Choice('pool_size', values=[2,3]), strides=1, padding="valid"))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(Spectral(units=hp.Int('units', min_value=1000, max_value=3000, step=500),**spectral_config, activation=activation))
    model.add(Dense(10, use_bias=use_bias, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[0.001,0.003,0.005,0.01,0.02,0.025,0.03,0.05])),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

if __name__=="__main__":
    d,c=40,1
    epochs = 20
    batch_size=100
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data=get_dataset(args=defaults, download=False, regenerate=True)
    x_train, y_train, x_test,y_test = data['x'], data['y'], data['x_test'],data['y_test']
    x_train, x_test = x_train.reshape(-1,d), x_test.reshape(-1,d)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        with tf.device("/device:GPU:0"):
            # Définir le tuner
            tuner = Hyperband(
                build_best_model,
                objective='val_accuracy',
                max_epochs=epochs,
                factor=3,
                directory='tuner/tuning',
                project_name='model'
            )
            # Regarder un récapitulatif de la recherche d'hyperparamètres
            tuner.search_space_summary()

            tuner.search(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_test, y_test))
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print(best_hps.values)
    else:
        raise Exception("No GPU available")