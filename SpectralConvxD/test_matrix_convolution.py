from utilsSimpleConv2D import *




def both_convolution(BATCH_SIZE, KERNEL_SIZES, SEQ_LENGTH, IN_CHANNELS, FILTERS, STRIDE,PADDING, INPUTS, TOL=1e-9):
    
    """
    Test the convolution function with the given parameters.
    Parameters:
    - BATCH_SIZE: int, size of the batch    
    - KERNEL_SIZES: int, size of the kernel
    - SEQ_LENGTH: int, length of the sequence
    - IN_CHANNELS: int, number of input channels
    - FILTERS: int, number of filters
    - STRIDE: int, stride for the convolution
    - PADDING: str, padding type ('SAME' or 'VALID')
    - INPUTS: tf.Tensor, input tensor for the convolution
    - TOL: float, tolerance for output comparison
    Returns:
    - bool: True if the test passes, False otherwise
    """
   
    # 1. Création de la convolution personnalisée
    custom_conv = matrix_conv_2d(
        filters=FILTERS,
        kernel_size=KERNEL_SIZES,
        strides=STRIDE,
        padding=PADDING,
        activation=None
    )
    custom_conv.build((BATCH_SIZE, SEQ_LENGTH,SEQ_LENGTH, IN_CHANNELS))
    
    # 2. Exécution de la convolution personnalisée
    outputs_1 = custom_conv.conv_jit(INPUTS)
    # 3. Récupération des poids de la convolution personnalisée
    weights = custom_conv.get_base()[0,: :, :]
    outputs_2=tf.constant(convolution(INPUTS[0,:,:,0], weights))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if tf.reduce_all(tf.abs(outputs_1[0,:,:,0] - outputs_2) < TOL).numpy():
            print("Outputs match!")
            return True
        else: 
            print('Test failed!')
            return False
    else:
        print("No GPU found, skipping test.")
        if tf.reduce_all(tf.abs(outputs_1[0,:,:,0] - outputs_2) < TOL).numpy():
            print("Outputs match!")
            return True
        else: 
            print('Test failed!')
            return False

def test_matrix_convolution():
    """
    Test the convolution function with the given parameters.
    
    Parameters:
    - BATCH_SIZE: int, size of the batch
    - KERNEL_SIZES: int, size of the kernel
    - SEQ_LENGTH: int, length of the sequence
    - IN_CHANNELS: int, number of input channels
    - FILTERS: int, number of filters
    - STRIDE: int, stride for the convolution
    - PADDING: str, padding type ('SAME' or 'VALID')
    - INPUTS: tf.Tensor, input tensor for the convolution
    - TOL: float, tolerance for output comparison
    
    Returns:
    - bool: True if the test passes, False otherwise
    """
    
    # Parameters for the convolution
    BATCH_SIZE = 1    
    KERNEL_SIZES = 3
    SEQ_LENGTH = 4
    IN_CHANNELS = 1
    FILTERS = 1
    STRIDE = 1
    PADDING= 'SAME'
    TOL= 1e-6
    INPUTS = tf.random.normal((BATCH_SIZE, SEQ_LENGTH,SEQ_LENGTH, IN_CHANNELS))
    # Run the test
    assert both_convolution(BATCH_SIZE=BATCH_SIZE,
                            KERNEL_SIZES=KERNEL_SIZES,
                            SEQ_LENGTH=SEQ_LENGTH,
                            IN_CHANNELS=IN_CHANNELS,
                            FILTERS=FILTERS,
                            STRIDE=STRIDE,
                            PADDING=PADDING,
                            INPUTS=INPUTS,
                            TOL=TOL)==True
    
if __name__=='__main__':
    print("Hi,world !")