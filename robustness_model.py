import SpectralConvxD as spc
import tensorflow as tf

if __name__ == "__main__":
    # Define the spectral configurations
    x_train, y_train,x_test, y_test=spc.generate_data(name_data='mnist1d')

    maxpooling_config ={'strides': 1,
                        'padding': 'valid' }

    hyperparameters = { 'filters' :20,
                    'input_shape' :(40,),
                    'learning_rate' : 0.01,
                    'epochs' : 20,
                    'batch_size' :100,
                    'activation': 'relu',
                    'labels' : 20,
                    'conxd':1,
                    'pool_size':4,
                    'full_training':True,
                    'pre_training': False,
                    }

    # hyperparameters
    models_pars,_= spc.pars()
    models_name=['specConvXd']
    replication = {'rep':5} 
    depth={'N':[2500]}
    drop={'p':[0,0.05,0.1,0.15,0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7, 0.75,0.8, 0.85,0.9,0.95,1]}
    # path =r"C:\Users\jketchak\Documents\DALPHAcommunity\UNAMUR\EUREKA\btwg_ml\SpectralConvxD\SpectralConvxD\DataWarehouse\dtest"
    # path=spc.utils.remplacer_backslash(path)
    path="/mnt/c/Users/jketchak/Documents/DALPHAcommunity/UNAMUR/EUREKA/btwg_ml/SpectralConvxD/SpectralConvxD/DataWarehouse/dtest"
    print(f"Path to save results: {path}")
    # Create the model
    models = spc.SpectralCnn(hyperparameters=hyperparameters,maxpooling_config=maxpooling_config)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        with tf.device("/device:GPU:0"):
            for name in models_name:
                for order in range(replication['rep']):
                    print(f"Replication {order+1}/{replication['rep']}")
                    for units in depth['N']:
                        print(f"units {units} :\n")
                        # compile, train and evaluate the model
                        models.compile_models(units=2500,spectral_config=models_pars.get(name).get('spectral_config'),
                                            spectral_cnn1d_config=models_pars.get(name).get('spectral_cnn1d_config'),
                                            spectral_cnn2d_config=models_pars.get(name).get('spectral_cnn2d_config'),
                                            name=name,
                                            layers_name=['convx','spec1','spec2'],
                                            layer_cible=None)
                        history=models.train(x_train, y_train, x_test, y_test, name=name,verbose=0)
                        for p in drop.get('p'):
                            models.evaluate(x_test, y_test, trainable_weights=None, path=path, name=name, order=order,layer_name='spec1',p=p,pre_pruning=False)
                    break
    else :
        raise ValueError("No GPU available")