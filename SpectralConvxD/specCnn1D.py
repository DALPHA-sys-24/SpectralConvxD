
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, initializers

from .utils import *


class SpecCnn1D(Layer):
    def __init__(self, filters,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    activation="relu",
                    use_lambda_out=False,
                    use_lambda_in=False,
                    use_bias=False,
                    trainable_phi=True,
                    bias_initializer='zeros',
                    phi_initializer="glorot_uniform",
                    lambda_in_initializer="glorot_uniform",
                    lambda_out_initializer="glorot_uniform"):

        super(SpecCnn1D,self).__init__()

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
        
        if self.padding<0:
            raise NotImplemented("self.padding<0 isn't supported")  
        
        if self.padding not in [0, 1]:
            raise ValueError("self.padding must be 0 : 'valid' or 1 : 'same'")
        
        if self.stride < 1:
            raise ValueError("self.stride must be >= 1")
        
        if self.stride >1 and self.padding==1:
            raise ValueError("self.stride > 1 and self.padding == 1 is not supported")  
        
        if self.kernel_size <= 1:
            raise ValueError("self.kernel_size must be >= 1")
        if self.kernel_size% 2 == 0:
            raise ValueError("self.kernel_size must be odd")
            
        if self.padding == 1 and self.stride==1:
            self.padding = math.floor((self.kernel_size-1) / 2)
            self.input_shape=input_shape[1]+2*self.padding
            self.output_shape: int = input_shape[1]
            self.pad=tf.pad(tf.linalg.diag(tf.ones((input_shape[1],),dtype=tf.float32)), paddings=[[self.padding , self.padding ], [0, 0]], mode='CONSTANT', constant_values=0)
            self.indices_phi()
        
        elif self.padding == 0 and self.stride>=1:
            self.input_shape=input_shape[1]
            self.output_shape: int = math.floor((self.input_shape + 2*self.padding- self.kernel_size) / self.stride) + 1
            self.pad=tf.linalg.diag(tf.ones((input_shape[1],),dtype=tf.float32))
            self.indices_phi()
        else:
            raise NotImplemented("Not implemented !")
            
        # Ensure that the number of indices matches the expected size for the convolution operation
        assert len(self.indices) == self.filters * self.output_shape * self.kernel_size
        # \phi
        if self.trainable_phi:
            self.kernel = self.add_weight(
                name='phi',
                shape=(self.filters,self.kernel_size),
                initializer=self.phi_initializer,
                dtype=tf.float32,
                trainable=self.trainable_phi)
        else:
            self.kernel = tf.ones((self.filters,self.kernel_size),dtype=tf.float32)
            
        # \lambda_in
        if self.use_lambda_in:
            self.lambda_in = self.add_weight(
                name='lambda_in',
                shape=(self.filters,self.input_shape),
                initializer=self.lambda_in_initializer,
                dtype=tf.float32,
                trainable=self.use_lambda_in)
        else:
            self.lambda_in = tf.ones((self.filters,self.input_shape),dtype=tf.float32)


        # \lambda_out
        if self.use_lambda_out:
            self.lambda_out = self.add_weight(
                name='lambda_out',
                shape=(self.filters,  self.output_shape),
                initializer=self.lambda_out_initializer,
                dtype=tf.float32,
                trainable=self.use_lambda_out)
        else:
            self.lambda_out = tf.zeros((self.filters,self.output_shape),dtype=tf.float32)
        

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
        # \weights of  phi -> flatten
        weights= tf.repeat(self.kernel, repeats=self.output_shape, axis=0, name=None)
        weights = tf.reshape(weights, shape=(self.filters * self.output_shape * self.kernel_size,))
        
        # \phi
        phi = tf.sparse.SparseTensor(
        indices=self.indices, values=weights,
        dense_shape=(self.filters,self.output_shape,self.input_shape))
        phi = tf.sparse.to_dense(phi)
        
        # \encode
        encode= tf.linalg.matmul(phi,tf.linalg.diag(self.lambda_in))
        
        # \decode
        decode= tf.linalg.matmul( tf.linalg.diag(self.lambda_out),phi)
        
        
        # \kernel
        kernel=encode-decode
        

        # \output      
        outputs=tf.linalg.matmul(self.pad,inputs,transpose_b=True)
        outputs =tf.matmul(a=kernel, b=outputs)
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
                    
if __name__ == "__main__": 
    print("SpecCnn1D Layer is ready to use.")   