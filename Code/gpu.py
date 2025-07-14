%load_ext autoreload
%autoreload 2
import time 
from models import *
from utilsSimpleConv2D import *


BATCH_SIZE = 100
SEQ_LENGTH = [3,40, 100, 128, 200, 500, 600, 700,750,10000]
IN_CHANNELS = 1
FILTERS = 20
KERNEL_SIZE = 3
STRIDE = 1
REPEAT = 50
INPUT_SHAPE=(BATCH_SIZE,SEQ_LENGTH*IN_CHANNELS)

TIMES={'classic_conv':[],'custom_conv':[]}

# 1. Convolution classique
classic_conv = tf.keras.layers.Conv1D(
    filters=FILTERS,
    kernel_size=KERNEL_SIZE,
    strides=STRIDE,
    padding="same",  # aligné avec padding=1 dans ta version
    activation='relu',
    use_bias=True
)



# 2. Ta convolution perso
custom_conv = matrix_conv_1d(
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=1,
        activation='relu',
        use_lambda_in=False,
        use_lambda_out=False,
        use_bias=True,
        trainable_phi=True
)
for seq_length in SEQ_LENGTH:
    # Donnée aléatoire
    x = tf.random.normal((BATCH_SIZE, seq_length, IN_CHANNELS))
    x_customize=tf.reshape(x,shape=(BATCH_SIZE, seq_length*IN_CHANNELS))
    custom_conv.build((BATCH_SIZE,seq_length*IN_CHANNELS))

    # 3. Benchmark classique
    start = time.time()
    for _ in range(REPEAT):
        _ = classic_conv(x)
    end = time.time()

    TIMES['classic_conv'].append(f"{(end - start)/REPEAT:.6f}")

    # 4. Benchmark custom
    start = time.time()
    for _ in range(REPEAT):
        _ = custom_conv.conv(x_customize)
    end = time.time()

    TIMES['custom_conv'].append(f"{(end - start)/REPEAT:.6f}")