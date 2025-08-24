from utilsSimpleConv2D import *
from models import *
import traceback

#parameters for plotting
BUDGET = 2500
models_name=['reference','specCnn1d','Dspec']
replication = {'rep':10}  # Number of replications
depth={'N':[50,100, 150, 200, 500, 800, 1000, 1500, 2000, 2500]}
drop={'p':[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

# Parameters for the model

spectral_config = {  'is_base_trainable': False,
                     'is_diag_start_trainable': False,
                     'is_diag_end_trainable': True,
                     'use_bias': True
                    }

spectral_cnn1d_config ={ 'kernel_size': 3,
                         'stride': 2,
                         'padding': 0,
                         'trainable_phi':False,
                         'use_lambda_out':False,
                         'use_lambda_in' : True,
                         'use_bias': True}

maxpooling_config ={ 'pool_size': 2,
                     'strides': 1,
                     'padding': 'valid' }

hyperparameters = { 'filters' :20,
                    'input_shape' :40,
                    'learning_rate' : 0.01,
                    'epochs' : 20,
                    'batch_size' :100,
                    'activation': 'relu',
                    'labels' : 10}
# Create the model
models = SpectralCnn(spectral_config=spectral_config,
                     spectral_cnn1d_config=spectral_cnn1d_config,
                     hyperparameters=hyperparameters,
                     maxpooling_config=maxpooling_config)


# Load the results
try:
    df = {name: extract_accuracy_and_save(replication, depth,drop=drop,name=name) for name in models_name}
except:
    print("Error loading results:")

# Percentage of the budget to use for each depth
if drop is None :
    for name in models_name:
        percentile =[models.number_of_parameters(name=name,units=value,pruning=0.7)/models.number_of_parameters(name='reference',units=BUDGET,pruning=0.7) for value in depth['N']]
        df[name]['p'] = percentile

# Plot the results
plot_results(df,
             x_min=0.1,
             x_max=1,
            y_min=0.1,
            y_max=1.0,
            xlabel='p',
            ylabel='accuracy',
            show_fig=True,
            use_grid=False,
            percentile=True,
            save_fig=True)

print("successfully plotted the results")