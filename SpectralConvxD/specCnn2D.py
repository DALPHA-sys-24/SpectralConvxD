from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, initializers

from .utils import *

class SpecCnn2D(Layer):

    def __init__(self, filters,
                 kernel_size=3,
                 strides=1,
                 padding='VALID',
                 use_lambda_out=False,
                 use_lambda_in=False,
                 use_encode=False,
                 use_decode=False,
                 trainable_omega_diag=True,
                 trainable_omega_triu=True,
                 trainable_omega_tril=True,
                 trainable_aggregate=True,
                 use_bias=False,
                 kernel_initializer="glorot_uniform",
                 pruning_initializer="glorot_uniform",
                 encoding_initializer="glorot_uniform",
                 activation="relu"):

        super(SpecCnn2D,self).__init__()

        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_size = kernel_size

        self.use_encode = use_encode
        self.use_decode = use_decode

        self.use_lambda_in = use_lambda_in
        self.use_lambda_out = use_lambda_out

        self.trainable_aggregate=trainable_aggregate
        self.trainable_omega_tril = trainable_omega_tril
        self.trainable_omega_triu = trainable_omega_triu
        self.trainable_omega_diag = trainable_omega_diag

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.pruning_initializer = initializers.get(pruning_initializer)
        self.encoding_initializer = initializers.get(encoding_initializer)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.size_omega_part = self.filters * self.kernel_size * math.floor((self.kernel_size - 1) / 2)
        self.pad = math.floor((self.kernel_size - 1) / 2)

        # -----------------------------------------matrix_pruning------------------------------------

        self.matrix_pruning = self.add_weight(
            name='matrix_pruning',
            shape=(input_shape[3],self.filters),
            initializer=self.pruning_initializer,
            dtype=tf.float32,
            trainable=self.trainable_aggregate)

        # -----------------------------------------matrix_pad------------------------------------
        if self.padding == "SAME":
            if self.strides > 1:
                raise Exception("Not implemented: paddind=SAME and strides>1. if padding=SAME, strides=1")

            # Right_shape
            self.Right_shape: Tuple = input_shape[1] + 2 * self.pad, input_shape[2] + 2 * self.pad
            inputShape = input_shape[1], input_shape[2]
            # matrix_pad
            self.matrix_pad = build_matrix_padding(input_shape=inputShape, pad=self.pad)

        elif self.padding == "VALID":
            # Right_shape
            self.Right_shape: Tuple = input_shape[1], input_shape[2]
            # matrix_pad
            self.matrix_pad = tf.constant(np.identity(input_shape[1] * input_shape[2]), dtype="float32")

        else:
            raise Exception("Padding not found")

        # -------------------------------------------------------------------------------------

        # ------------------------------out_in_shape_phi_indices-------------------------------
        self.set_indices_phi()
        self.set_indices_ul_triangular()
        # -------------------------------------------------------------------------------------

        # \omega_diag
        if self.trainable_omega_diag:
            self.omega_diag = self.add_weight(
                name='omega_diag',
                shape=(self.filters, self.kernel_size,),
                initializer=self.kernel_initializer,
                dtype=tf.float32,
                trainable=self.trainable_omega_diag)

        else:
            self.omega_diag = tf.constant(
                np.random.uniform(size=(self.filters, self.kernel_size,), low=-0.05, high=0.05), dtype=tf.float32,
                name='omega_diag')

        # \omega_triu
        if self.trainable_omega_triu:
            self.omega_triu = self.add_weight(
                name='omega_triu',
                shape=(1, self.size_omega_part),
                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None),
                dtype=tf.float32,
                trainable=self.trainable_omega_triu)

        else:
            self.omega_triu = tf.constant(np.random.uniform(size=(1, self.size_omega_part), low=-0.05, high=0.05),
                                          dtype=tf.float32, name='omega_triu')

        # \omega_tril
        if self.trainable_omega_tril:
            self.omega_tril = self.add_weight(
                name='omega_tril',
                shape=(1, self.size_omega_part),
                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None),
                dtype=tf.float32,
                trainable=self.trainable_omega_tril)

        else:
            self.omega_tril = tf.constant(np.random.uniform(size=(1, self.size_omega_part), low=-0.05, high=0.05),
                                          dtype=tf.float32, name='omega_tril')

        # ---------------------------------indeice_Omega:part-----------------------------------
        self.set_indices_ul_triangular()
        # -------------------------------------------------------------------------------------

        # \use_lambda_in
        if self.use_lambda_in:
            self.use_lambda_in = self.add_weight(name='use_lambda_in',
                                                 shape=(self.kernel_size, 1),
                                                 initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05,
                                                                                           seed=None),
                                                 dtype=tf.float32,
                                                 trainable=self.use_lambda_in)

        else:
            self.use_lambda_in = tf.random.uniform(shape=(self.kernel_size, 1), minval=-0.05, maxval=0.05,
                                                   dtype=tf.float32, name='use_lambda_in')

        # \use_lambda_out
        if self.use_lambda_out:
            self.use_lambda_out = self.add_weight(name='use_lambda_out',
                                                  shape=(1, self.kernel_size),
                                                  initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05,
                                                                                            seed=None),
                                                  dtype=tf.float32,
                                                  trainable=self.use_lambda_out)
        else:
            self.use_lambda_out = tf.random.uniform(shape=(1, self.kernel_size), minval=-0.05, maxval=0.05,
                                                    dtype=tf.float32, name='use_lambda_out')

        # use_encode
        if self.use_encode:
            self.use_encode = self.add_weight(name='use_encode',
                                              shape=(1, self.Right_shape[0] * self.Right_shape[1]),
                                              initializer=tf.ones_initializer(),
                                              dtype=tf.float32,
                                              trainable=self.use_encode)
        else:

            self.use_encode = tf.zeros(shape=(1, self.Right_shape[0] * self.Right_shape[1]), dtype=tf.float32,
                                      name='use_encode')

        # use_decode
        if self.use_decode:
            self.use_decode = self.add_weight(name='use_decode',
                                              shape=(self.output_lenght, 1),
                                              initializer=self.encoding_initializer ,
                                              dtype=tf.float32,
                                              trainable=self.use_decode)

        else:
            self.use_decode = tf.ones(shape=(self.output_lenght, 1), dtype=tf.float32, name='use_decode')

        # --------------------------------------------bias---------------------------------------
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                dtype=tf.float32,
                trainable=self.use_bias)
        else:
            self.bias = None
        # ---------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------
        self.build = True
        # ---------------------------------------------------------------------------------------

    def call(self, inputs):

        # ----------------------------------Inputs---------------------------------------------


        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        flatten =tf.matmul(flatten, self.matrix_pruning)
        upFlatten = tf.matmul(a=self.matrix_pad, b=flatten)
        upFlatten=tf.transpose(upFlatten, perm=[2, 0, 1])



        # ----------------------------------\Omega---------------------------------------------
        omega_high = tf.sparse.SparseTensor(
            indices=self.indices_triu, values=self.omega_triu[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_lower = tf.sparse.SparseTensor(
            indices=self.indices_tril, values=self.omega_tril[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_high = tf.sparse.to_dense(omega_high)
        omega_lower = tf.sparse.to_dense(omega_lower)
        omega = omega_lower + tf.linalg.diag(self.omega_diag, k=0) + omega_high

        # ----------------------------------Base-----------------------------------------------

        base = tf.multiply(omega, self.use_lambda_in - self.use_lambda_out)

        # ------------------------------Build Noyau--------------------------------------------

        kernel = tf.reshape(base, shape=(self.filters, self.kernel_size * self.kernel_size))

        kernel = tf.repeat(kernel, repeats=self.output_lenght, axis=0, name=None)

        kernel = tf.reshape(kernel, shape=(-1, self.filters * self.output_lenght * self.kernel_size * self.kernel_size))

        kernel = tf.sparse.SparseTensor(
            indices=self.indices, values=kernel[0],
            dense_shape=(self.filters, self.output_lenght, self.Right_shape[0] * self.Right_shape[1])
        )

        kernel = tf.sparse.to_dense(kernel)

        kernel = tf.linalg.matmul(kernel, tf.linalg.diag(self.use_encode[0, :], k=0)) - tf.linalg.matmul(
            tf.linalg.diag(self.use_decode[:, 0], k=0), kernel)

        kernel = tf.transpose(kernel, perm=[0, 2, 1])

        # -----------------------------------Outputs-------------------------------------------------
        outputs = tf.matmul(a=upFlatten, b=kernel)
        outputs = tf.transpose(outputs, perm=[1, 2, 0])
        outputs = tf.reshape(outputs, shape=(-1, self.out_shape1, self.out_shape2, self.filters))

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs

    def set_indices_phi(self, *args):
        self.indices: List[Tuple] = list()

        self.out_shape1: int = math.floor((self.Right_shape[0] - self.kernel_size) / self.strides) + 1
        self.out_shape2: int = math.floor((self.Right_shape[1] - self.kernel_size) / self.strides) + 1
        self.output_lenght: int = self.out_shape1 * self.out_shape2

        for filters in range(self.filters):
            count: int = 1
            shift: int = 0
            for i in range(self.output_lenght):
                if i == count * (self.out_shape2):
                    count += 1
                    shift += self.kernel_size + (self.strides - 1) * self.Right_shape[1]
                else:
                    if shift:
                        shift += self.strides
                    else:
                        shift += 1
                for block in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        self.indices.append((filters, i, block * self.Right_shape[1] + shift - 1 + j))

    def set_indices_ul_triangular(self, *args):
        self.indices_triu: List[Tuple] = list()
        self.indices_tril: List[Tuple] = list()

        for filters in range(self.filters):
            for i in range(1, self.kernel_size):
                for j in range(i):
                    self.indices_tril.append((filters, i, j))

        for filters in range(self.filters):
            for i in range(self.kernel_size):
                for j in range(i + 1, self.kernel_size):
                    self.indices_triu.append((filters, i, j))

    def get_indices_phi(self, *args):
        return self.indices

    def get_base(self, *args):
        # ----------------------------------\Omega---------------------------------------------
        omega_high = tf.sparse.SparseTensor(
            indices=self.indices_triu, values=self.omega_triu[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_lower = tf.sparse.SparseTensor(
            indices=self.indices_tril, values=self.omega_tril[0],
            dense_shape=(self.filters, self.kernel_size, self.kernel_size)
        )
        omega_high = tf.sparse.to_dense(omega_high)
        omega_lower = tf.sparse.to_dense(omega_lower)
        omega = omega_lower + tf.linalg.diag(self.omega_diag, k=0) + omega_high
        # ----------------------------------Base-----------------------------------------------

        base = tf.multiply(omega, self.use_lambda_in - self.use_lambda_out)
        return base


if __name__ == '__main__':
    print("SpecCnn2d Layer is ready to use.")   