import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
import SpectralConvxD as spc
print("SpectralConvxD version :{}".format(spc.__version__))


def build_best_model(hp):
    d=40
    activation="relu"
    
    spectral_config = { 'is_base_trainable': True,
                    'is_diag_start_trainable': False,
                    'is_diag_end_trainable': False,
                    'use_bias': True
                    }
    spectral_cnn1d_config={ 'kernel_size':3,
                        'padding': 0,
                        'trainable_phi':False,
                        'use_lambda_out':False,
                        'use_lambda_in' : True,
                        'use_bias': True,
                        'activation':"relu"
                     }

    model= tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(d,)))
    model.add(spc.SpecCnn1D(filters=20, stride=1,**spectral_cnn1d_config))
    
    model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Choice('pool_size', values=[2,3,4,5]), strides=1, padding="valid"))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(spc.Spectral(units=hp.Int('units', min_value=1000, max_value=3000, step=500),**spectral_config, activation=activation))
    spectral_config['is_diag_end_trainable'] =False
    model.add(spc.Spectral(10, **spectral_config, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[0.001,0.003,0.005,0.01,0.02,0.025,0.03,0.05])),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

def spec_tuner(epochs,batch_size):
    x_train, y_train,x_test, y_test=spc.generate_data(name_data='mnist1d')

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
        print("No GPU found. Running on CPU.")
        with tf.device("/device:CPU:0"):
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

if __name__ == "__main__":
    spec_tuner(epochs=20, batch_size=100)