import time 
from models import *
from utilsSimpleConv2D import *



def get_times(KERNEL_SIZES,STRIDE, FILTERS,SEQ_LENGTH, BATCH_SIZE, IN_CHANNELS, REPEAT):
    """
    measure the time taken to run the model on random input data
    """
    TIMES = {
                'seq_length': [],
                'kernel_size': [],
                'variant': [],
                'med': [],
                'q1': [],
                'q2': []
            }

    for kernel_size in KERNEL_SIZES:
        # 1. Création des couches avec la taille de filtre variable
        classic_conv = tf.keras.layers.Conv1D(
            filters=FILTERS,
            kernel_size=kernel_size,
            strides=STRIDE,
            padding="same",
            use_bias=False,
            activation='relu'
        )

        custom_conv = matrix_conv_1d(
            kernel_size=kernel_size,
            stride=STRIDE,
            padding=1,  # correspond à padding="same"
            activation='relu',
            use_lambda_in=False,
            use_lambda_out=False,
            trainable_phi=True
        )

        for seq_length in SEQ_LENGTH:
            # Donnée aléatoire
            x = tf.random.normal((BATCH_SIZE, seq_length, IN_CHANNELS))
            x_customize = tf.reshape(x, shape=(BATCH_SIZE, seq_length * IN_CHANNELS))
            custom_conv.build((BATCH_SIZE, seq_length * IN_CHANNELS))
            classic_conv.build((BATCH_SIZE,seq_length,IN_CHANNELS))

            # Benchmark convolution personnalisée
            custom_times = []
            for _ in range(REPEAT):
                start = time.time()
                _ = custom_conv.conv_jit(x_customize)
                end = time.time()
                custom_times.append(end - start)

            # Statistiques custom
            TIMES['seq_length'].append(seq_length)
            TIMES['kernel_size'].append(kernel_size)
            TIMES['variant'].append('custom_conv')
            TIMES['med'].append(np.median(custom_times))
            TIMES['q1'].append(np.quantile(custom_times, 0.25))
            TIMES['q2'].append(np.quantile(custom_times, 0.85))

            # Benchmark convolution classique
            classic_times = []
            for _ in range(REPEAT):
                start = time.time()
                _ = classic_conv(x)
                end = time.time()
                classic_times.append(end - start)

            # Statistiques classiques
            TIMES['seq_length'].append(seq_length)
            TIMES['kernel_size'].append(kernel_size)
            TIMES['variant'].append('classic_conv')
            TIMES['med'].append(np.median(classic_times))
            TIMES['q1'].append(np.quantile(classic_times, 0.25))
            TIMES['q2'].append(np.quantile(classic_times, 0.85))

    # Résultats dans un DataFrame
    data = pd.DataFrame(TIMES)
    # data.set_index('kernel_size', inplace=True)
    # data = data.pivot_table(index='kernel_size', columns='variant', values=['med', 'q1', 'q2'])
    # data.set_index('seq_length', inplace=True)
    return data

if __name__ == '__main__':
    import tensorflow as tf
    tf.keras.backend.clear_session()
    print("Running benchmarks for matrix_conv_1d...")
    BATCH_SIZE = 1
    KERNEL_SIZES = [3, 9, 15, 21, 27, 31] 
    SEQ_LENGTH = [50, 100, 128, 200, 500, 600, 700,750,1000,1500,2000,2500,3000,3500,4000,4500,5000]
    IN_CHANNELS = 1
    FILTERS = 1
    STRIDE = 1
    REPEAT = 200


    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        with tf.device("/device:GPU:0"):
            df_times= get_times(SEQ_LENGTH=SEQ_LENGTH, BATCH_SIZE=BATCH_SIZE, KERNEL_SIZES=KERNEL_SIZES,STRIDE=STRIDE, FILTERS=FILTERS,IN_CHANNELS=IN_CHANNELS, REPEAT=REPEAT)
    else:
        raise ValueError("No GPU available")
    # 3. Sauvegarde des résultats
    df_times.to_csv('benchmarks_matrix_conv_1d_gpu.csv', index=False)   
    # 4. Visualisation
    plot_benchmarks(df_times, fontsize=12, alpha=0.2, loc='center', dpi=300, use_grid=False, show_now=False, save_fig=False, filename='benchmarks_conv1d_gpu.pdf')
    print("Benchmarks completed and saved to 'benchmarks_matrix_conv_1d_gpu.csv' and 'benchmarks_conv1d_gpu.pdf'.")
    # End of file Benchmarks_matrix_conv_1d.py
    # End of file Benchmarks_matrix_conv_1d.p
