import SpectralConvxD as spc
import tensorflow as tf


x_train, y_train,x_test, y_test=spc.generate_data(name_data='mnist1d')


maxpooling_config ={'strides': 1,
                     'padding': 'valid' }

hyperparameters = { 'filters' :20,
                    'input_shape' :(40,),
                    'learning_rate' : 0.025,
                    'epochs' : 20,
                    'batch_size' :100,
                    'activation': 'relu',
                    'labels' : 20,
                    'conxd':1,
                    'pool_size':5,
                    'full_training':False,
                    'pre_training': False,
                    }

replication = {'rep':5} 
depth={'N':[50,100, 150, 200, 500, 800, 1000, 1500, 2000, 2500,3000]}
drop={'p':[0,0.05,0.1,0.15,0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7, 0.75,0.8, 0.85,0.9,0.95,1]}
path =r"/mnt/c/Users/jketchak/Documents/DALPHAcommunity/UNAMUR/EUREKA/btwg_ml/SpectralConvxD/SpectralConvxD/DataWarehouse/New"
path=spc.utils.remplacer_backslash(path)
models_pars,models_name= spc.pars()


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    with tf.device("/device:GPU:0"):
        for name in models_name:
            print(f"Model name: {name}")
            for order in range(replication['rep']):
                print(f"Replication {order+1}/{replication['rep']}")
                for units in depth['N']:
                    print(f"units {units} :\n")
                    models = spc.SpectralCnn(hyperparameters=hyperparameters,maxpooling_config=maxpooling_config)
                    models.compile_models(units=units,spectral_config=models_pars.get(name).get('spectral_config'),
                                        spectral_cnn1d_config=models_pars.get(name).get('spectral_cnn1d_config'),
                                        spectral_cnn2d_config=models_pars.get(name).get('spectral_cnn2d_config'),
                                        name=name,
                                        layers_name=['convx','spec1','spec2'],
                                        layer_cible=None)
                    models.train(x_train, y_train, x_test, y_test, name=name,verbose=0)
                    models.evaluate(x_test, y_test, trainable_weights=None, path=path, name=name, order=order,layer_name='spec1',p=-1,pre_pruning=True,save_accuracy=True)
else :
    raise ValueError("No GPU available")