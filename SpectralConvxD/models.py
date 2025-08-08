from .utils import *
from .SpectralLayer import *
from .specCnn1D import *
from .specCnn2D import *
import copy
import tensorflow as tf 
import numpy as np 



class SpectralCnn(object):
    def __init__(self,hyperparameters,maxpooling_config=None):
        self.maxpooling_config=maxpooling_config
        self.hyperparameters=hyperparameters
        self.percentage_zeroed=[]
        self.cut_value=[]
        self.trainable_weigths={}
        
    def compile_models(self,units:int= 1000,spectral_config=None,spectral_cnn1d_config=None,spectral_cnn2d_config=None, name:str=None,layers_name=None):
        """_summary_

        Args:
            units (int, optional): _description_. Defaults to 1000.
            spectral_config (_type_, optional): _description_. Defaults to None.
            spectral_cnn1d_config (_type_, optional): _description_. Defaults to None.
            spectral_cnn2d_config (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to [reference,Dspec,specConvXd] 

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        
        self.name=name
        self.layers_name=layers_name
        if len(self.layers_name)!=3:
            raise ValueError("layers_name must have exactly 3 elements")
        if self.name is None:
            raise ValueError("model's name must be defined")
        if not isinstance(units, int):
            raise ValueError("hyperparameters must be a dictionary")
        
        if self.hyperparameters.get('use_pruning') and self.name=='specConvXd':
            print("Pruning: Ranking the Nodes Based on the Eigenvalues")
            spectral_config['is_base_trainable']=True
            spectral_cnn1d_config['use_lambda_in']=False
            spectral_cnn1d_config['trainable_phi']=True
            
        elif  self.hyperparameters.get('use_pruning') and self.name=='reference':
            print("Pruning: Absolute value of the incoming connectivity collapse")
        
        elif self.hyperparameters.get('use_pruning')==False and self.name in ['reference','specConvXd']:
            print("Pruning: Pershaps pre-training")
        
        else:
            raise ValueError("No pruning for this model")
            
            
        self.set_spectral_config(spectral_config)
        self.set_spectral_cnn1d_config(spectral_cnn1d_config)
        self.set_spectral_cnn2d_config(spectral_cnn2d_config)
        
        
        if self.hyperparameters.get('pre_training') and self.name=='specConvXd':
            self. spectral_cnn1d_config['use_lambda_in']=False
            self.spectral_cnn1d_config['trainable_phi']=True 

        if  self.name in  ['reference','Dspec','specConvXd']:
            self.model= tf.keras.Sequential()
            self.model.add(tf.keras.layers.Input(shape=self.hyperparameters.get('input_shape')))
            if self.hyperparameters.get('conxd')==1:
                self.model.add(SpecCnn1D(filters=self.hyperparameters.get('filters'), **self.spectral_cnn1d_config,activation=self.hyperparameters.get('activation'),name=self.layers_name[0]))
                self.model.add(tf.keras.layers.MaxPooling1D(pool_size=self.hyperparameters.get('pool_size'),**self.maxpooling_config))
            elif self.hyperparameters.get('conxd')==2:
                self.model.add(SpecCnn2D(filters=self.hyperparameters.get('filters'), **self.spectral_cnn2d_config,activation=self.hyperparameters.get('activation'),name=self.layers_name[0]))
                self.model.add(tf.keras.layers.MaxPooling2D(pool_size=self.hyperparameters.get('pool_size'),**self.maxpooling_config))
            else:
                raise ValueError("conxd's value must be 1 or 2")

            self.model.add(tf.keras.layers.Flatten())

            self.model.add(Spectral(units,**self.spectral_config, activation=self.hyperparameters.get('activation'),name=self.layers_name[1]))
            
            
            if self.hyperparameters.get('pre_training') and self.name=='specConvXd':
                print("Pre training: Spectral robustness")
                self.spectral_config['is_base_trainable']=True
                self.spectral_config['is_diag_end_trainable']=False
            
            self.model.add(Spectral(self.hyperparameters.get('labels'), **self.spectral_config, activation='softmax',name=self.layers_name[2]))
            
            opt = tf.keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate')) 
            self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        else:
            raise ValueError("model's name must be 'reference' ,'Dspec',or 'specConvXd'")
        
        self.hyperparameters['units']=units
    
    def train(self,x_train, y_train, x_test=None, y_test=None,name=None,verbose=0):
        if name is None:
            raise ValueError("model's name must be defined")   
        if name != 'reference' and name != 'Dspec' and name != 'specConvXd':
            raise ValueError("model's name must be 'reference', 'specConvXd', or 'Dspec'")
        if not isinstance(verbose, int):
            raise ValueError("verbose must be an integer") 

        if self.name != name:
            raise ValueError("model's name must be the same as the one used in compile_models()")
        if x_train is None or y_train is None:
            raise ValueError("x_train and y_train must be defined")
        
        if x_test is None or y_test is None:
            history = self.model.fit(x_train, y_train, batch_size=self.hyperparameters.get('batch_size'),
                                     epochs=self.hyperparameters.get('epochs'), verbose=1)
        else:
            history = self.model.fit(x_train, y_train, batch_size=self.hyperparameters.get('batch_size'),
                                     epochs=self.hyperparameters.get('epochs'), verbose=verbose,
                                     validation_data=(x_test, y_test))
            
        self.history=history
        self.trainable_weigths={name:[var for var in self.model.get_layer(name=name).trainable_variables] for name in self.layers_name}
    
    def evaluate(self,x_test, y_test, path=None, name=None, order=0,layers=None,p=0):
        """Evaluate the model on the test data.

        Args:
            x_test: Test input data.
            y_test: Test target data.
            path: Path to save accuracy results.
            name: Model name.
            order: Order/index for saving results.
            layers: List of layer indices for weight operations.
            p: Pruning/robustness parameter.

        Returns:
            Accuracy value if applicable.
        """
        
        if name is None:
            raise ValueError("model's name must be defined")
        if name != 'reference' and name != 'specConvXd' and name != 'Dspec':
            raise ValueError("model's name must be 'reference', 'specConvXd' or 'Dspec'")
        if name!=self.name:
            raise ValueError("model's name must be the same as the one used in compile_models()")
        if x_test is None or y_test is None:
            raise ValueError("x_test and y_test must be defined")
        
        if order <0 :
            raise NotImplemented("Fatal error")
        if p<0 or p>1:
            raise NotImplemented("Fatal error")
        
        if 0 <= p and p <= 1 and 0<= order:
            self.p_robustness(layers,p,name)
            accuracy=self.model.evaluate(x_test, y_test, batch_size=self.hyperparameters.get('batch_size'), verbose="auto")[1]
            with open(f"{path}/{name}/accuracy{p}_{order}.txt", "a+") as f:
                f.write(str(accuracy))
                f.write("\n")
            return accuracy
        else:
            raise NotImplemented("Fatal Error")
            

    
    def p_robustness(self,layers: List= [3],p=0,name=None,w=None):
        if name is None:
            raise ValueError("model's name must be defined")
        if name=='reference':
            for k,layer in enumerate(layers):
                if w is None:
                    weigths=copy.deepcopy(self.weigths[k])
                else:
                    weigths=w
                # units=weigths[0].shape[1]
                result, collapsed_indices, row_sums, threshold_value=self.absolute_value_of_the_incoming_connectivity_collapse(weigths[0], p)
                assert result.shape==weigths[0].shape
                weigths[0]=result
                # indices=random.sample(range(units),int(units*p))
                # weigths[0][:,indices]=self.mask_by_quantile(weigths[0][:,indices],1)
                self.model.layers[layer].set_weights(weigths)
        else: 
            for k,layer in enumerate(layers):
                if w is None:
                    weigths=copy.deepcopy(self.weigths[k])
                else:
                    weigths=w
                # weigths[1]=self.mask_by_quantile(weigths[1], p)
                cut_value = self.percentile_robust(weigths[1][0,:], p)
                self.cut_value.append(cut_value)
                pruning_weights, percentage_zeroed = self.threshold_filter(cut_value, weigths[1][0,:])
                self.percentage_zeroed.append(percentage_zeroed)
                assert pruning_weights.shape==weigths[1][0,:].shape
                weigths[1][0,:] = pruning_weights
                self.model.layers[layer].set_weights(weigths)
    
    def absolute_value_of_the_incoming_connectivity_collapse(self,matrix, p_percent):
        """
        Calculates the sum of absolute values of weights for each row, orders them,
        and sets to zero the p percent lowest values (weakest connections).
        
        Args:
            matrix (list or np.array): Input matrix
            p_percent (float): Percentage of weakest rows to collapse (0.0-1.0)
        
        Returns:
            tuple: (filtered_matrix, collapsed_indices, row_sums, threshold_value)
                - filtered_matrix (np.array): Matrix with weakest rows set to zero
                - collapsed_indices (np.array): Indices of rows that were collapsed
                - row_sums (np.array): Array of absolute row sums before collapse
                - threshold_value (float): The threshold value used for collapse
        """
        # Validate percentage
        if not isinstance(p_percent, (int, float)):
            raise TypeError("p_percent must be a number")
        if not (0 <= p_percent <= 1):
            raise ValueError("p_percent must be between 0 and 1")
        
        # Convert to numpy array if not already
        if isinstance(matrix, np.ndarray):
            mat=matrix.copy()

        if mat.size == 0:
            return mat, np.array([]), np.array([]), 0.0
        
        # Create a copy to avoid modifying the original
        result = mat.copy()
        
        # Calculate sum of absolute values for each row
        row_sums = np.sum(np.abs(mat), axis=1)
        
        # Calculate how many rows to collapse
        num_rows = mat.shape[0]
        num_to_collapse = int(np.ceil(num_rows * p_percent))
        
        if num_to_collapse == 0:
            # No rows to collapse
            return result, np.array([]), row_sums, 0.0
        
        # Find indices of rows with smallest sums (weakest connections)
        sorted_indices = np.argsort(row_sums)
        collapsed_indices = sorted_indices[:num_to_collapse]
        
        # Determine threshold value
        if num_to_collapse < num_rows:
            threshold_value = row_sums[sorted_indices[num_to_collapse - 1]]
        else:
            threshold_value = row_sums[sorted_indices[-1]]
        
        # Set the weakest rows to zero
        result[collapsed_indices, :] = 0
        
        return result, collapsed_indices, row_sums, threshold_value

    
    
    
    def percentile_robust(self, array, p) -> float:
        """
        Calculate the percentile of an array, ignoring NaN values.
        If the input is empty or contains only NaN values, raise a ValueError.

        Parameters:
        - array: numpy array or list containing the data.   
        - p: float, the percentile to compute (between 0 and 1).

        Returns:      
        - The percentile value.

        Raises:
        - ValueError: if p is not between 0 and 1, or if the array is empty or contains only NaN values.
        """
        if not isinstance(p, (int, float)):
            raise ValueError("p must be a number")
        if isinstance(array, list):
            array = np.array(array)
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if array.size == 0:
            raise ValueError("Input array is empty")

        # Remove NaN values
        array_clean = array[~np.isnan(array)]
        if array_clean.size == 0:
            raise ValueError("Input array contains only NaN values")

        return np.percentile(array_clean, p * 100)


    def threshold_filter(self, threshold, vector):
        """
        Sets to zero the values in the vector that are smaller than the threshold.
        
        Args:
            threshold (float): Threshold value
            vector (list or np.array): Input vector
        
        Returns:
            tuple: (filtered_vector, percentage_zeroed)
                - filtered_vector (np.array): Vector with values < threshold set to zero
                - percentage_zeroed (float): Percentage of values that were set to zero
        """
        
        if not isinstance(vector, (list, np.ndarray)):
            raise ValueError("Vector must be a list or a numpy array")
        if isinstance(vector, list):
            vector = np.array(vector)
        if vector.size == 0:
            raise ValueError("Input vector is empty")
        if np.isnan(threshold):
            raise ValueError("Threshold cannot be NaN")
        if np.isnan(vector).all():
            raise ValueError("Input vector contains only NaN values")

        
        # Create a copy to avoid modifying the original
        result = vector.copy()
        
        # Count values below threshold
        values_below_threshold = np.sum(vector < threshold)
        total_values = len(vector)
        
        # Calculate percentage
        percentage_zeroed = (values_below_threshold / total_values) * 100 if total_values > 0 else 0
        
        # Set values below threshold to zero
        result[result < threshold] = 0
        
        return result, percentage_zeroed

  
    def summary(self,name):
        if name is None:
            raise ValueError("model's name must be defined")
        if name=='reference':
            print(f"Reference model summary:\n ")
            self.model.summary()
        elif name=='specConvXd': 
            print("specConvXd model summary:\n ")
            self.model.summary()
        elif name=='Dspec':
            print("Dspec model summary:\n ")
            self.model.summary()
        else:
            raise ValueError("model's name must be 'reference','specConvXd' or 'Dspec'")
        

    def mask_by_quantile(self, tensor, percentage:float):
        # Flatten the tensor if it's multidimensional
        original_shape = tf.shape(tensor)
        flat = tf.reshape(tensor, [-1])

        n_total = tf.cast(tf.size(flat), dtype=tf.float32)
        k = tf.cast(tf.math.floor(tf.multiply(n_total, percentage)), dtype=tf.int32)
        n_total = tf.cast(tf.size(flat), dtype=tf.int32)

        # Get sorted indices (from largest to smallest)
        sorted_indices = tf.argsort(flat, direction='DESCENDING')
        
        # Keep the top (n - k) indices
        indices_to_keep = sorted_indices[:n_total - k]

        # Create a boolean mask: True to keep, False to zero
        mask = tf.scatter_nd(
            indices=tf.expand_dims(indices_to_keep, 1),
            updates=tf.ones_like(indices_to_keep, dtype=tf.bool),
            shape=tf.shape(flat)
        )

        # Apply the mask: zero out the rest
        modified_flat = tf.where(mask, flat, tf.zeros_like(flat))

        # Reshape back to original shape
        return tf.reshape(modified_flat, original_shape)
    
        
    def get_trainable_weigths(self,layers_name=None):
        return self.trainable_weights

    def set_trainable_weigths(self,trainable_weights=None):
        self.trainable_weights=trainable_weights
    
    def get_maxpooling_config(self):
        return self.maxpooling_config
    
    def get_spectral_config(self):
        return self.spectral_config
    
    def get_spectral_cnn1d_config(self):
        return self.spectral_cnn1d_config
    
    def get_spectral_cnn2d_config(self):
        return self.spectral_cnn2d_config
     
    def get_hyperparameters(self):
        return self.hyperparameters
     
    def set_spectral_config(self,spectral_config):
        self.spectral_config=spectral_config
        
    def set_spectral_cnn1d_config(self,spectral_cnn1d_config):
        self.spectral_cnn1d_config=spectral_cnn1d_config
        
    def set_spectral_cnn2d_config(self,spectral_cnn2d_config):
        self.spectral_cnn2d_config=spectral_cnn2d_config
        
    def set_maxpooling_config(self,maxpooling_config):
        self.maxpooling_config=maxpooling_config
    
    def set_hyperparameters(self,hyperparameters):
        if not isinstance(hyperparameters, dict):
            raise ValueError("hyperparameters must be a dictionary")
        self.hyperparameters=hyperparameters
    
    def number_of_parameters(self,name='reference',units=1000,pruning=None):
        if pruning is None:
            raise ValueError("pruning must be defined")
        if not isinstance(name, str):
            raise ValueError("model's name must be a string")
        if name is None:
            raise ValueError("model's name must be defined")
        if units is None:
            raise ValueError("units must be defined")
        if not isinstance(units, int):
            raise ValueError("units must be an integer")
        if name is None:
            raise ValueError("model's name must be defined")
        if name not in ['reference', 'specConvXd', 'Dspec']:
            raise ValueError("model's name must be 'reference', 'specCnn1d' or 'Dspec'")
        if units <= 0:
            raise ValueError("units must be greater than 0")
        
        if name == 'reference':
            try:
                output_shape = math.floor((self.hyperparameters.get('input_shape') + 2*self.spectral_cnn1d_config.get("padding") - self.spectral_cnn1d_config.get('kernel_size'))/self.spectral_cnn1d_config.get('stride')) + 1
                output_shape = math.floor((output_shape - self.maxpooling_config.get('pool_size'))/self.maxpooling_config.get('strides')) + 1
            except KeyError or TypeError:
                raise ValueError("kernel_size, padding, and stride must be defined in spectral_cnn1d_config")
            
            try:
                weights=self.spectral_cnn1d_config.get('kernel_size') * self.hyperparameters.get('filters')+ units*(self.hyperparameters.get('labels') + output_shape*self.hyperparameters.get('filters'))
            except KeyError or TypeError:
                raise ValueError("kernel_size, filters, labels, and output_shape must be defined in hyperparameters")
            bias=self.hyperparameters.get('filters') + units + self.hyperparameters.get('labels')
            value = weights + bias
            return value
        elif name == 'Dspec':
            try:
                output_shape = math.floor((self.hyperparameters.get('input_shape') + 2*self.spectral_cnn1d_config.get("padding") - self.spectral_cnn1d_config.get('kernel_size'))/self.spectral_cnn1d_config.get('stride')) + 1
                output_shape = math.floor((output_shape - self.maxpooling_config.get('pool_size'))/self.maxpooling_config.get('strides')) + 1
            except KeyError or TypeError:
                raise ValueError("kernel_size, padding, and stride must be defined in spectral_cnn1d_config")
            
            try:
                weights=self.hyperparameters.get('input_shape') * self.hyperparameters.get('filters')+ units*(self.hyperparameters.get('labels') + output_shape*self.hyperparameters.get('filters'))
            except KeyError or TypeError:
                raise ValueError("kernel_size, filters, labels, and output_shape must be defined in hyperparameters")
            bias=self.hyperparameters.get('filters') + units + self.hyperparameters.get('labels')
            value = weights + bias
            return value
        
        
        elif name == 'specConvXd':
            print("number_of_parameters: specConvXd , X=1 \n")
            try:
                weights=self.hyperparameters.get('input_shape') * self.hyperparameters.get('filters') + units + self.hyperparameters.get('labels')
            except KeyError or TypeError:
                raise ValueError("kernel_size, filters, labels, and output_shape must be defined in hyperparameters")
            bias=self.hyperparameters.get('filters') + units + self.hyperparameters.get('labels')
            value = weights + bias
            return value
        else:
            raise ValueError("Unknown model name")

if __name__ == "__main__":
    print("This is a module, not a script. Please import it in your code.")

