import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import threading

from tensorflow import keras
from utilsSimpleConv2D import*
from SpectralLayer import*
from fetch_data import*
from keras_tuner import Hyperband
from mnist1d.data import make_dataset, get_dataset_args,get_dataset

                
class SpecCnn1d_v01(Layer):
    def __init__(self, filters,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    activation="relu",
                    use_bias=False,
                    trainable_phi=True,
                    bias_initializer='zeros',
                    phi_initializer="glorot_uniform"):

        super(SpecCnn1d_v01,self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        
        self.trainable_phi = trainable_phi
        self.activation = activations.get(activation)
        self.bias_initializer = initializers.get(bias_initializer)
        self.phi_initializer = initializers.get(phi_initializer)

    def build(self, input_shape):
        if self.padding>0:
            raise NotImplemented("self.padding>0 isn't supported")
        
        self.output_shape: int = math.floor((input_shape[1] + 2*self.padding- self.kernel_size) / self.stride) + 1
        self.input_shape=input_shape[1]
        self.indices_phi()
        assert len(self.indices)==self.filters*self.output_shape*self.kernel_size

        # \phi
        if self.trainable_phi:
            self.kernel = self.add_weight(
                name='phi',
                shape=(self.filters,self.kernel_size,),
                initializer=self.phi_initializer,
                dtype=tf.float32,
                trainable=self.trainable_phi)
        else:
            self.kernel = tf.constant(np.ones(self.filters,self.kernel_size),'phi')

        # \bias
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                dtype=tf.float32,
                trainable=self.use_bias)
        else:
            self.bias = None


    def call(self, inputs):
        # \weights of  phi
        weights= tf.repeat(self.kernel, repeats=self.output_shape, axis=0, name=None)
        #  \flatten
        weights = tf.reshape(weights, shape=(self.filters * self.output_shape * self.kernel_size,))
        # \phi
        phi = tf.sparse.SparseTensor(
        indices=self.indices, values=weights,
        dense_shape=(self.filters,self.output_shape,self.input_shape))
        phi = tf.sparse.to_dense(phi)
        
        outputs =tf.matmul(a=phi, b=inputs,transpose_b=True)
        outputs=tf.transpose(outputs,perm=[2,1,0])

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs

    def indices_phi(self,*args):
        self.indices: List[Tuple] = list()
        for f in range(self.filters):
            for i in range(self.output_shape):
                for j in range(i*self.stride,i*self.stride+self.kernel_size):
                    self.indices.append((f,i,j))


class SpecCnn1d_v02(Layer):
    def __init__(self, filters,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 activation="relu",
                 use_lambda_out=True,
                 use_lambda_in=False,
                 use_bias=False,
                 trainable_phi=False,
                 bias_initializer='zeros',
                 phi_initializer="glorot_uniform",
                 lambda_in_initializer="glorot_uniform",
                 lambda_out_initializer="glorot_uniform"):

        super(SpecCnn1d_v02, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.use_lambda_in = use_lambda_in
        self.use_lambda_out = use_lambda_out

        self.trainable_phi = trainable_phi
        self.activation = activations.get(activation)
        self.bias_initializer = initializers.get(bias_initializer)
        self.phi_initializer = initializers.get(phi_initializer)
        self.lambda_in_initializer = initializers.get(lambda_in_initializer)
        self.lambda_out_initializer = initializers.get(lambda_out_initializer)

    def build(self, input_shape):
        if self.padding > 0:
            raise NotImplemented("self.padding>0 isn't supported")

        self.input_shape = input_shape[1]
        self.output_shape: int = math.floor((self.input_shape + 2 * self.padding - self.kernel_size) / self.stride) + 1
        self.indices_phi()
        assert len(self.indices) == self.filters * self.output_shape * self.kernel_size

        # \phi
        if self.trainable_phi:
            self.kernel = self.add_weight(
                name='phi',
                shape=(self.filters, self.kernel_size),
                initializer=self.phi_initializer,
                dtype=tf.float32,
                trainable=self.trainable_phi)
        else:
            self.kernel = tf.constant(np.ones((self.filters, self.kernel_size)), dtype=tf.float32) / self.kernel_size

        # \lambda_in
        if self.use_lambda_in:
            self.lambda_in = self.add_weight(
                name='lambda_in',
                shape=(self.filters, self.input_shape),
                initializer=self.lambda_in_initializer,
                dtype=tf.float32,
                trainable=self.use_lambda_in)
        else:
            self.lambda_in = tf.constant(np.ones((self.filters, self.input_shape)),
                                         dtype=tf.float32)

        # \lambda_out
        if self.use_lambda_out:
            self.lambda_out = self.add_weight(
                name='lambda_out',
                shape=(self.filters, self.output_shape),
                initializer=self.lambda_out_initializer,
                dtype=tf.float32,
                trainable=self.use_lambda_out)
        else:
            self.lambda_out = tf.constant(np.zeros((self.filters, self.output_shape)),
                                          dtype=tf.float32)

        # \bias
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                dtype=tf.float32,
                trainable=self.use_bias)
        else:
            self.bias = None

        self.build = True

    def call(self, inputs):
        # \weights of  phi -> flatten
        weights = tf.repeat(self.kernel, repeats=self.output_shape, axis=0, name=None)
        weights = tf.reshape(weights, shape=(self.filters * self.output_shape * self.kernel_size,))

        # \phi
        phi = tf.sparse.SparseTensor(
            indices=self.indices, values=weights,
            dense_shape=(self.filters, self.output_shape, self.input_shape))
        phi = tf.sparse.to_dense(phi)

        # \encode
        encode = tf.linalg.matmul(phi, tf.linalg.diag(self.lambda_in))

        # \decode
        decode = tf.linalg.matmul(tf.linalg.diag(self.lambda_out), phi)

        # \kernel
        kernel = encode - decode

        # \output
        outputs = tf.matmul(a=kernel, b=inputs, transpose_b=True)
        outputs = tf.transpose(outputs, perm=[2, 1, 0])

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs

    def indices_phi(self, *args):
        self.indices: List[Tuple] = list()
        for f in range(self.filters):
            for i in range(self.output_shape):
                for j in range(i * self.stride, i * self.stride + self.kernel_size):
                    self.indices.append((f, i, j))

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
    model.add(SpecCnn1d_v02(filters=20,kernel_size=3,stride=hp.Choice('stride', values=[1,2,3]),padding=0,trainable_phi=True,use_lambda_out=False,use_lambda_in=False,activation=activation,use_bias=use_bias))
    
    model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Choice('pool_size', values=[2,3]), strides=1, padding="valid"))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(Spectral(units=hp.Int('units', min_value=1000, max_value=3000, step=500),**spectral_config, activation=activation))
    # model.add(Spectral(10, **spectral_config, activation='softmax'))
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