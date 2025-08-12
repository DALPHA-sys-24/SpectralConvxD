import SpectralConvxD as spc
import numpy as np
import tensorflow as tf
import copy 
from typing import Any


if __name__ == "__main__":
    # Define the spectral configurations
    x_train, y_train,x_test, y_test=spc.generate_data(name_data='mnist1d')

    maxpooling_config ={'strides': 1,
                        'padding': 'valid' }

    hyperparameters = { 'filters' :20,
                    'input_shape' :(40,),
                    'learning_rate' : 0.03,
                    'epochs' : 20,
                    'batch_size' :100,
                    'activation': 'relu',
                    'labels' : 20,
                    'conxd':1,
                    'pool_size':2,
                    'full_training':True,
                    'pre_training': True,
                    }

    # hyperparameters
    replication = {'rep':5} 
    depth={'N':[2500]}
    drop={'p':[0,0.05,0.1,0.15,0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7, 0.75,0.8, 0.85,0.9,0.95,1]}
    # path =r"C:\Users\jketchak\Documents\DALPHAcommunity\UNAMUR\EUREKA\btwg_ml\SpectralConvxD\SpectralConvxD\DataWarehouse\dtest"
    # path=spc.utils.remplacer_backslash(path)
    path="/mnt/c/Users/jketchak/Documents/DALPHAcommunity/UNAMUR/EUREKA/btwg_ml/SpectralConvxD/SpectralConvxD/DataWarehouse/dtest"
    
    print(f"Path to save results: {path}")





    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for p in drop.get('p'):
            for order in range(replication['rep']):
                # Pre-training weights diag_end
                inv=2
                hyperparameters['pre_training']=True
                hyperparameters['full_training']=False
                models_pars,models_name= spc.pars()
                models = spc.SpectralCnn(hyperparameters=hyperparameters,maxpooling_config=maxpooling_config)
                models.compile_models(units=2500,spectral_config=models_pars.get(models_name[inv]).get('spectral_config'),
                                    spectral_cnn1d_config=models_pars.get(models_name[inv]).get('spectral_cnn1d_config'),
                                    spectral_cnn2d_config=models_pars.get(models_name[inv]).get('spectral_cnn2d_config'),
                                    name=models_name[inv],
                                    layers_name=['convx','spec1','spec2'],
                                    layer_cible=None)
                models.train(x_train, y_train, x_test, y_test, name=models_name[inv],verbose=0)
                print(f"Pre-training weights diag_end for {models_name[inv]} with p={p}")

                pre_taining_weights=models.percentile_spectral_filter(trainable_weights=models.model.get_layer(name='spec1').variables,
                                                                    name=models_name[inv],
                                                                    p=p,
                                                                    layer_name='spec1')
                #get diag_end weights
                for k,var in enumerate(pre_taining_weights):
                    if var.name=='diag_end':
                        inv=0
                        hyperparameters['pre_training']=False
                        hyperparameters['full_training']=False
                        diag_end=var.numpy()
                        break

                #Post-training weights diag_end
                del models
                models = spc.SpectralCnn(hyperparameters=hyperparameters,maxpooling_config=maxpooling_config)
                models.compile_models(units=2500,spectral_config=models_pars.get(models_name[inv]).get('spectral_config'),
                                    spectral_cnn1d_config=models_pars.get(models_name[inv]).get('spectral_cnn1d_config'),
                                    spectral_cnn2d_config=models_pars.get(models_name[inv]).get('spectral_cnn2d_config'),
                                    name=models_name[inv],
                                    layers_name=['convx','spec1','spec2'],
                                    layer_cible='spec1')

                # Set pre-training weights
                old_weights=models.model.get_layer(name='spec1').variables
                for k,var in enumerate(old_weights):
                    if var.name=='diag_end':
                        old_weights[k].assign(diag_end)
                        break
                models.model.get_layer(name='spec1').set_weights(old_weights)
                models.train(x_train, y_train, x_test, y_test, name=models_name[inv],verbose=0)
                print(f"Post-training weights diag_end for {models_name[inv]} with p={p}")
                models.evaluate(x_test, y_test, trainable_weights=None, path=path, name=models_name[inv], order=order,layer_name='spec1',p=p,pre_pruning=True)
