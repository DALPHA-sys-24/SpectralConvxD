from models import *


#DATASET = 'mnist1d'
x_train, y_train,x_test, y_test=generate_data(color_mode='grayscale',seed=23,subset="both",validation_split=0.2,name_data='mnist')
#PARAMETERS

spectral_config = {  'is_base_trainable': True,
                     'is_diag_start_trainable': False,
                     'is_diag_end_trainable': False,
                     'use_bias': True
                    }

spectral_cnn1d_config ={ 'kernel_size': 3,
                         'stride': 2,
                         'padding': 0,
                         'trainable_phi':False,
                         'use_lambda_out':False,
                         'use_lambda_in' : True,
                         'use_bias': True}

spectral_cnn2d_config={ 'kernel_size': 3,
                        'strides': 1,
                        'padding': 'VALID',
                        "use_lambda_out":False,
                        "use_lambda_in":False,
                        "use_encode":False,
                        "use_decode":False,
                        'use_bias': False,
                        "trainable_omega_diag":True,
                        "trainable_omega_triu":True,
                        "trainable_omega_tril":True,
                        "kernel_initializer":"glorot_uniform"
                        }

maxpooling_config ={ 'pool_size': (2,2),
                     'strides': 2,
                     'padding': 'valid' }

hyperparameters = { 'filters' :2,
                    'input_shape' :(28,28,1),
                    'learning_rate' : 0.01,
                    'epochs' : 20,
                    'batch_size' :100,
                    'activation': 'relu',
                    'labels' : 10}




# Create the model
models = SpectralCnn(spectral_config=spectral_config,
                     spectral_cnn1d_config=spectral_cnn1d_config,
                     spectral_cnn2d_config=spectral_cnn2d_config,
                     hyperparameters=hyperparameters,
                     maxpooling_config=maxpooling_config)

# Compile the model
# extract the number of parameters
# Set the number of replications and the depth of the model
#models_name=['pruningDspec','Dspec','specCnn1d','reference']
models_name=['reference','Dspec']
replication = {'rep':10}  # Number of replications
#depth={'N':[50,100, 150, 200, 500, 800, 1000, 1500, 2000, 2500]}
depth={'N':[2000]}
drop={'p':[0.0,0.03,0.05,0.07,0.09,0.1,0.199,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.83,0.85,0.87,0.89, 0.9, 1]}
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    with tf.device("/device:GPU:0"):
        for name in models_name:
            for order in range(replication['rep']):
                print(f"Replication {order+1}/{replication['rep']}")
                for units in depth['N']:
                    print(f"units {units} :\n")
                    # compile, train and evaluate the model
                    models.compile_models(units=units,Use_pruning=False,pruning=0,use_base_and_Lambda=True, name=name)
                    history=models.train(x_train, y_train, x_test=x_test, y_test=y_test, name=name,verbose=0,layers=[3])
                    for p in drop.get('p'):
                        models.evaluate(x_test, y_test,order=order, name=name,layers=[3],p=p)
            #Just the first one
else :
    raise ValueError("No GPU available")