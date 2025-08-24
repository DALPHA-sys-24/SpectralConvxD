
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple,List,Dict
from matplotlib import pyplot as plt
from tensorflow.keras import activations, initializers
from .fetch_data import *


def pars():
    models_pars={'reference':
                    {"spectral_config":{'is_base_trainable': True,
                                        'is_diag_start_trainable': False,
                                        'is_diag_end_trainable': False,
                                        'use_bias': True
                                        },
                     "spectral_cnn1d_config":{ 'kernel_size': 3,
                                                'stride': 3,
                                                'padding': 0,
                                                'trainable_phi':True,
                                                'use_lambda_out':False,
                                                'use_lambda_in' : False,
                                                'use_bias': True
                                              },
                     "spectral_cnn2d_config":{ 'kernel_size': 3,
                                                'strides': 1,
                                                'padding': 'VALID',
                                                "use_lambda_out":False,
                                                "use_lambda_in":False,
                                                "use_encode":False,
                                                "use_decode":False,
                                                'use_bias': True,
                                                "trainable_omega_diag":True,
                                                "trainable_omega_triu":True,
                                                "trainable_omega_tril":True,
                                                "trainable_aggregate":False,
                                                }
                    },
                    
             'Dspec':{"spectral_config":{ 'is_base_trainable': True,
                                        'is_diag_start_trainable': False,
                                        'is_diag_end_trainable': False,
                                        'use_bias': True
                                        },
                     "spectral_cnn1d_config":{ 'kernel_size': 3,
                                                'stride': 1,
                                                'padding': 0,
                                                'trainable_phi':False,
                                                'use_lambda_out':False,
                                                'use_lambda_in' : True,
                                                'use_bias': True
                                              },
                     "spectral_cnn2d_config":{ 'kernel_size': 3,
                                                'strides': 1,
                                                'padding': 'VALID',
                                                "use_lambda_out":False,
                                                "use_lambda_in":False,
                                                "use_encode":False,
                                                "use_decode":True,
                                                'use_bias': True,
                                                "trainable_omega_diag":False,
                                                "trainable_omega_triu":False,
                                                "trainable_omega_tril":False,
                                                "trainable_aggregate":False,
                                                }
                    },
             
             'specConvXd':{"spectral_config":{ 'is_base_trainable': False,
                                        'is_diag_start_trainable': False,
                                        'is_diag_end_trainable': True,
                                        'use_bias': True
                                        },
                     "spectral_cnn1d_config":{ 'kernel_size': 3,
                                                'stride': 3,
                                                'padding': 0,
                                                'trainable_phi':False,
                                                'use_lambda_out':False,
                                                'use_lambda_in' : True,
                                                'use_bias': True
                                              },
                     "spectral_cnn2d_config":{ 'kernel_size': 3,
                                                'strides': 1,
                                                'padding': 'VALID',
                                                "use_lambda_out":False,
                                                "use_lambda_in":False,
                                                "use_encode":False,
                                                "use_decode":True,
                                                'use_bias': True,
                                                "trainable_omega_diag":False,
                                                "trainable_omega_triu":False,
                                                "trainable_omega_tril":False,
                                                "trainable_aggregate":False,
                                                }
                    }
            }
    return models_pars,list(models_pars.keys())

def remplacer_backslash(chaine):
    """
    Remplace tous les backslashes (\) par des slashes (/) dans une chaîne.
    
    Args:
        chaine (str): La chaîne de caractères à modifier
    
    Returns:
        str: La chaîne avec les backslashes remplacés par des slashes
    """
    return chaine.replace('\\', '/')

def get_accuracy(name_file="SpecConv2dClassique",i=1,PATH="C:/Users/jketchak/OneDrive - Université de Namur/Bureau/DALPHAcommunity/UNAMUR/MATHESES/PUBLICATIONS/DONNEES/Convolution Neural Networks in the spectral"):
    """_summary_

    Args:
        name_file (str, optional): _description_. Defaults to "SpecConv2dClassique".
        i (int, optional): _description_. Defaults to 1.
        PATH (str, optional): _description_. Defaults to "C:/Users/jketchak/OneDrive - Université de Namur/Bureau/DALPHAcommunity/UNAMUR/MATHESES/PUBLICATIONS/DONNEES/Convolution Neural Networks in the spectral".

    Returns:
        _type_: _description_
    """
    with open(f"{PATH}/Data{i}/{name_file}/{name_file}.csv", "r") as f:
        data=pd.read_csv(f, sep=",", header=0)
    try:
        data.set_index('N', inplace=True)
    except:
        data.set_index('S', inplace=True)
    return data


def extract_accuracy_and_save(replication, depth,drop=None,name=None,path=None):
    """_summary_

    Args:
        replication (_type_): _description_
        depth (_type_): _description_
        drop (_type_, optional): _description_. Defaults to None.
        name (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    if drop is None:
        accuracy:Dict[int,pd.DataFrame]=dict()
        if name is None:
            raise ValueError("Name must be provided for the dataset.")

        for order in range(replication['rep']):
            with open(f"{path}/{name}/accuracy{order}.txt", "r") as f:
                accuracy[order]=pd.read_csv(f, sep=" ", header=None)
                accuracy[order].rename(columns={0:f"r{order+1}"},inplace=True)
        data=accuracy[0]
        for order in range(1,replication['rep']):
            data = pd.concat([data,accuracy[order]], axis=1)
        
        data['N']=depth['N']
        data.set_index('N', inplace=True)

        data["max"]=data.apply(max,axis=1)
        data["min"]=data.apply(min,axis=1)
        data["mean"]=data.apply(np.mean,axis=1)
        data["q1"]=data[[f"r{order+1}" for order in  range(replication['rep']) ]].quantile(0.25,axis=1)
        data["q2"]=data[[f"r{order+1}" for order in  range(replication['rep']) ]].quantile(0.75,axis=1)
        data["med"]=data[[f"r{order+1}" for order in  range(replication['rep']) ]].median(axis=1)
        data["sigma"]= data[[f"r{order+1}" for order in  range(replication['rep']) ]].std(axis=1)
        data.to_csv(f"{path}/{name}/{name}.csv", header=True, index=True)
    
    else:           
        accuracy:Dict[int,Dict[int,pd.DataFrame]]={i: {} for i in range(len(drop.get('p')))}
        for k,p in enumerate(drop.get('p')):
            for order in range(replication['rep']):
                with open(f"{path}/{name}/accuracy{p}_{order}.txt", "r") as f:
                    accuracy[k][order]=pd.read_csv(f, sep=" ", header=None)
        
                            
                    
        data=pd.DataFrame({p: [value.loc[0,0] for value in accuracy[k].values()] for k,p in  enumerate(drop.get('p'))}).T
        
        data['p']=drop.get('p')
        data["max"]=data.apply(max,axis=1)
        data["min"]=data.apply(min,axis=1)
        data["q1"]=data[[order for order in  range(replication['rep'])]].quantile(0.25,axis=1)
        data["q2"]=data[[order for order in  range(replication['rep']) ]].quantile(0.75,axis=1)
        data["med"]=data[[order for order in  range(replication['rep']) ]].median(axis=1)
        data["sigma"]= data[[order for order in  range(replication['rep'])]].std(axis=1)
        data.to_csv(f"{path}/{name}/{name}.csv", header=True, index=True)
    return data


def plot_results(df,x_min,x_max,y_min,y_max,alpha=0.1,lw=2,figsize=(9,5),dpi=250,loc='lower right',markersize=2,linestyle='--',fontsize=12,use_grid=False,xlabel='N',ylabel='Accuracy',name_fig=None,save_fig=True,show_fig=False,percentile=True):
    """
    This function is used to plot the results of the experiments.
    It extracts the accuracy from the results and saves them in a dictionary.
    
    Parameters  
    ----------

    df: DataFrame   
        DataFrame containing the results of the experiments with columns: 'p', 'med', 'q1', 'q2'.  
    x_min: int
        Minimum value for the x-axis.
    x_max: int  
        Maximum value for the x-axis.
    y_min: float
        Minimum value for the y-axis.
    y_max: float
        Maximum value for the y-axis.
    alpha: float    
        Transparency level for the shaded area between quantiles.
    lw: int
        Line width for the plot.
    figsize: Tuple[int, int]    
        Size of the figure in inches (width, height).
    dpi: int
        Dots per inch for the figure resolution.    
    loc: str    
        Location of the legend in the plot.
    markersize: int
        Size of the markers in the plot.
    linestyle: str  
        Style of the line in the plot (e.g., '--', '-').
    fontsize: int
        Font size for the plot labels.  
    xlabel: str 
        Label for the x-axis.
    ylabel: str 
        Label for the y-axis.
    name_fig: str   
        Name of the figure to save. If None, the figure will not be saved.
    save_fig: bool
        If True, the plot will be saved to a file.
    show_fig: bool
        If True, the plot will be displayed immediately. If False, it will be closed after saving.
    percentile: bool
        If True, the plot will use percentiles (q1, q2) for the y-axis. If False, it will use the index of the DataFrame.
    Returns
    -------
    None    
    
    This function does not return any value. It generates a plot and optionally saves it to a file.

    """
    # Color map for different models
    COLORS=["k","b","r","c","y","b","m"]
    MARKER=['*','o','^','v']
    
    fig, ax = plt.subplots(num=1,figsize=figsize,dpi=dpi)
    
    
    
    if percentile:
        ax.axis([x_min, x_max, y_min, y_max])
        for i, name in enumerate(df.keys()):
            ax.plot(df[name].p, df[name].med, lw=lw, label=name,linestyle=linestyle, color=COLORS[i],marker=MARKER[i],markersize=markersize)
            ax.fill_between(df[name].p, df[name].q1, df[name].q2, color=COLORS[i], alpha=alpha)
    else:
        ax.axis([x_min, x_max, y_min, y_max])
        for i, name in enumerate(df.keys()):
            ax.plot(df[name].index, df[name].med, lw=lw, label=name, color=COLORS[i])
            ax.fill_between(df[name].index, df[name].q1, df[name].q2, color=COLORS[i], alpha=alpha)
    if loc is not None:
        ax.legend(loc=loc)
        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
    plt.grid(use_grid)
    
    if save_fig:
        if percentile:
            plt.savefig(f'{name_fig}.pdf', dpi=dpi)
        else:   
            plt.savefig(f'{name_fig}.pdf', dpi=dpi)
    if show_fig:
        plt.show()
    plt.close(fig)



def plot_benchmarks(data, fontsize=12,alpha=0.2,loc='upper left', dpi=300, use_grid=False,show_now=True, save_fig=True, filename='benchmarks_matrix_conv_1d_gpu.pdf'): 
    """
    
    data: DataFrame
        DataFrame containing the benchmark results with columns: 'seq_length', 'kernel_size', 'variant', 'med', 'q1', 'q2'.
    fontsize: int  
        Font size for the plot labels.
    alpha: float
        Transparency level for the shaded area between quantiles.
    loc: str
        Location of the legend in the plot. 
    dpi: int
        Dots per inch for the figure resolution.   
    use_grid: bool
        Whether to display a grid in the plot.
    show_now: bool
        If True, the plot will be displayed immediately. If False, it will be closed after saving.
    save_fig: bool
        If True, the plot will be saved to a file. 
    filename: str
        Name of the file to save the plot. The file format will be inferred from the extension.
    Returns
    ------- 
    None
    ----------

    This function generates a plot comparing the performance of different convolution variants (classic and custom) across various sequence lengths and kernel sizes.       
    
    """
    fig, ax = plt.subplots(num=1,figsize=(12,7),dpi=dpi)
    COLORS = {'classic_conv': ['k', 'r', 'r', 'c', 'b', 'm', 'y', 'w'], 'custom_conv':['k', 'r', 'r', 'c', 'b', 'm', 'y', 'w']}
    LINESTYLE={'classic_conv': '-','custom_conv': '--'  }

    for c,kernel_size in enumerate(data['kernel_size'].unique()):
        if kernel_size >9:
            break
        # Filtrer les données pour le kernel_size actuel
        for name in data['variant'].unique():
            plt.semilogy(data['seq_length'].unique(),data.groupby('kernel_size').get_group(kernel_size).groupby('variant').get_group(name).med,
                        color=COLORS[name][c],
                        label=f'{name}-kernel_size-{kernel_size}',
                        linestyle=LINESTYLE[name])
            
            plt.fill_between(data['seq_length'].unique(),
                            data.groupby('kernel_size').get_group(kernel_size).groupby('variant').get_group(name).q1,
                            data.groupby('kernel_size').get_group(kernel_size).groupby('variant').get_group(name).q2,
                            color=COLORS[name][c], alpha=alpha,
                            linestyle=LINESTYLE[name])
            
    ax.legend(loc=loc)
    ax.set_xlabel('SEQ_LENGTH',fontsize=fontsize)
    ax.set_ylabel('Times (seconds)',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(use_grid)
    if save_fig:
        plt.savefig(f'{filename}', dpi=dpi)
    if show_now:
        plt.show()
    else:
        plt.close(fig)


def indices_phi(filters: int, N: int, M: int, F: int = 3, S: int = 1, *args):
    """_summary_

    Args:
        filters (int): _description_
        N (int): _description_
        M (int): _description_
        F (int, optional): _description_. Defaults to 3.
        S (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    indices: List[Tuple] = list()
    out_shape1: int = math.floor((N - F) / S) + 1
    out_shape2: int = math.floor((M - F) / S) + 1
    output_lenght: int = out_shape1 * out_shape2

    for filter in range(filters):
        count: int = 1
        shift: int = 0
        for i in range(output_lenght):
            if i == count * (out_shape2):
                count += 1
                shift += F + (S - 1) * M
            else:
                if shift:
                    shift += S
                else:
                    shift += 1
            for block in range(F):
                for j in range(F):
                    indices.append((filter, i, block * M + shift - 1 + j))
    return indices

def Build_J(N:int,M:int,F:int=3,S:int=1,*args):
    """_summary_

    Args:
        N (int): _description_
        M (int): _description_
        F (int, optional): _description_. Defaults to 3.
        S (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    out_shape1:int=math.floor((N-F)/S)+1
    out_shape2:int=math.floor((M-F)/S)+1
        
    row = tf.Variable(np.zeros(shape=(M,1)), dtype=tf.float32, trainable=False)
    row[0,0].assign(1)
    for j in range(out_shape2):
        for k in range(F):
            try:
                new_line=shift_(row,k,axis=0)
                J1 = tf.concat([J1, new_line], 1)
            except:
                J1 = row
        row=shift_(row, S,axis=0)
        
    del new_line
    col = tf.Variable(np.zeros(shape=(1,N)), dtype=tf.float32, trainable=False)
    col[0,0].assign(1)
    
    for i in range(out_shape1):
        for l in range(F):
            try:
                new_line=shift_(col,l,axis=1)
                J2 = tf.concat([J2, new_line], 0)
            except:
                J2 = col
        col=shift_(col, S,axis=1)
        
    return J1, J2

def build_matrix_strides(out_shape,strides):
    """_summary_

    Args:
        out_shape (_type_): _description_
        strides (_type_): _description_

    Returns:
        _type_: _description_
    """

    height = math.floor(out_shape[1] / strides)
    width = math.floor(out_shape[2] / strides)
    count = 1
    line = tf.Variable(np.zeros(shape=(out_shape[1] * out_shape[2])), dtype=tf.float32, trainable=False)
    line[0].assign(1)
    line = tf.reshape(line, shape=(1, line.shape[0]))
    S = line
    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                continue
            position = strides * (i * out_shape[2] + j)
            new_line = shift_(line, position)
            S = tf.concat([S, new_line], 0)
            count += 1
    return S


def build_matrix_padding(input_shape: Tuple, pad: int):
    """_summary_

    Args:
        input_shape (Tuple): _description_
        pad (int): _description_

    Returns:
        _type_: _description_
    """
    # block
    out_shape: Tuple = input_shape[0] + 2 * pad, input_shape[1] + 2 * pad
    width_matrix_padding: int = input_shape[0] * input_shape[1]
    height_matrix_padding: int = out_shape[0] * out_shape[1]

    size_block1: Tuple = out_shape[1], width_matrix_padding
    block1 = tf.zeros(shape=size_block1)
    size_block2: Tuple = pad, width_matrix_padding
    block2 = tf.zeros(shape=size_block2)

    line = tf.Variable(np.zeros(shape=(width_matrix_padding)), dtype=tf.float32, trainable=False)
    line[0].assign(1)
    line = tf.reshape(line, shape=(1, line.shape[0]))
    # initialisation
    M = line
    new_line = line
    matrix = block1

    for i in range(1, out_shape[0] - 1):
        matrix = tf.concat([matrix, block2], 0)

        for j in range(1, input_shape[1]):
            new_line = shift_(new_line, 1)
            M = tf.concat([M, new_line], 0)
        matrix = tf.concat([matrix, M], 0)
        matrix = tf.concat([matrix, block2], 0)
        del M
        new_line = shift_(new_line, 1)
        M = new_line

    matrix = tf.concat([matrix, block1], 0)
    assert matrix.shape == (height_matrix_padding, width_matrix_padding)
    return matrix


def convolution_at(img: np.ndarray,kernel: np.ndarray,i:int,j:int)->float:
    """_summary_

    Args:
        img (np.ndarray): _description_
        kernel (np.ndarray): _description_
        i (int): _description_
        j (int): _description_

    Returns:
        float: _description_
    """
    output = 0
    kernel_shape:Tuple =kernel.shape
    img_shape: Tuple = img.shape
    center_point:int =math.floor((kernel_shape[0]-1)/2)
    height=i-center_point
    width=j-center_point
    for s in range(height,i+center_point+1):
        for r in range(width, j+ center_point+1):
            if (s<0 or s>img_shape[0]-1 or r>img_shape[1]-1 or r<0):continue
            output+=img[s,r]*kernel[s-height,r-width]
    return output

def convolution(img: np.ndarray,kernel: np.ndarray)->np.ndarray:
    """_summary_

    Args:
        img (np.ndarray): _description_
        kernel (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    img_shape:Tuple =img.shape
    output =np.zeros(shape=img_shape,dtype="float32")
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            output[i,j]=convolution_at(img=img,kernel=kernel,i=i,j=j)
    return output

class matrix_conv_1d(object):
    """_summary_

    Args:
        object (_type_): _description_
    """
    def __init__(self,  kernel_size=3,
                        stride=1,
                        padding=0,
                        activation="relu",
                        use_lambda_out=False,
                        use_lambda_in=False,
                        trainable_phi=True,
                        phi_initializer="glorot_uniform",
                        lambda_in_initializer="glorot_uniform",
                        lambda_out_initializer="glorot_uniform"):


        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_lambda_in = use_lambda_in
        self.use_lambda_out = use_lambda_out
        
        self.trainable_phi = trainable_phi
        self.activation = activations.get(activation)
        self.phi_initializer = initializers.get(phi_initializer)
        self.lambda_in_initializer = initializers.get(lambda_in_initializer)
        self.lambda_out_initializer = initializers.get(lambda_out_initializer)
    
    def build(self, input_shape:Tuple)->None:
        
        if self.padding<0:
            raise NotImplemented("self.padding<0 isn't supported")  

        if self.padding> 1:
            self.padding = 1
            
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
        assert len(self.indices) == self.output_shape * self.kernel_size
        assert len(self.indices_in) == self.input_shape
        assert len(self.indices_out) == self.output_shape
        
        # \kernel phi
        if self.trainable_phi:
            self.kernel = tf.random.uniform(shape=(self.kernel_size,),minval=-0.5,maxval=0.5)
            
        else:
            self.kernel = tf.ones((self.kernel_size,),dtype=tf.float32)
        
        # \phi
        self.phi = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices, values= tf.reshape(tf.tile(tf.expand_dims(self.kernel, axis=0), [self.output_shape, 1]) , shape=(self.output_shape * self.kernel_size,)),
        dense_shape=(self.output_shape,self.input_shape)))
            
        # \lambda_in
        if self.use_lambda_in:
            lambda_in = tf.random.uniform(shape=(self.input_shape,),minval=-0.5,maxval=0.5)
        else:
            lambda_in = tf.ones((self.input_shape,),dtype=tf.float32)
            
        self.lambda_in =  tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices_in, values= lambda_in,dense_shape=(self.input_shape,self.input_shape)))

        # \lambda_out
        if self.use_lambda_out:
            lambda_out = tf.random.uniform(shape=(self.output_shape,),minval=-0.5,maxval=0.5)
        else:
            lambda_out = tf.zeros((self.output_shape,),dtype=tf.float32)
            
        self.lambda_out = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices_out, values=lambda_out,dense_shape=(self.output_shape,self.output_shape)))
        
        self.custom = True
        
    
    def conv(self, inputs: tf.Tensor):
        if not self.custom:
            raise ValueError("The layer has not been built yet.")

        # encode = phi @ lambda_in
        encode = tf.matmul(self.phi, self.lambda_in)

        # decode = phi^T @ lambda_out
        decode = tf.matmul( self.lambda_out,self.phi)
        
        # kernel = encode - decode
        kernel = encode - decode

        # apply padding then matmul
        outputs = tf.matmul(self.pad, inputs, transpose_b=True)  
        outputs = tf.matmul(kernel, outputs)                     
        outputs = tf.transpose(outputs)                          

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
    
    @tf.function(jit_compile=True)
    def conv_jit(self, inputs: tf.Tensor):
         return self.conv(inputs)

    def indices_phi(self,*args):
        self.indices: List[Tuple] = list()
        self.indices_in: List[Tuple] = list()
        self.indices_out: List[Tuple] = list()
        for i in range(self.output_shape):
            self.indices_out.append((i,i))
            for j in range(i*self.stride,i*self.stride+self.kernel_size):
                self.indices.append((i,j))        
        for i in range(self.input_shape):
            self.indices_in.append((i,i))


class matrix_conv_2d(object):
    """_summary_

    Args:
        object (_type_): _description_
    """
    def __init__(self, filters=1,
                 kernel_size=3,
                 strides=1,
                 padding='VALID',
                 use_encode=False,
                 use_decode=False,
                 activation=None):


        self.custom = False
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.use_encode = use_encode
        self.use_decode = use_decode
        self.activation = activations.get(activation)

    def build(self, input_shape):
        
        if self.padding not in ["SAME", "VALID"]:
            raise ValueError("Padding must be 'SAME' or 'VALID'.")
        if self.strides < 1:
            raise ValueError("Strides must be >= 1.")
        if self.strides > 1 and self.padding == "SAME":
            raise ValueError("Not implemented: padding='SAME' and strides>1. If padding='SAME', strides=1.")
        if self.kernel_size <= 1:
            raise ValueError("Kernel size must be >= 1.")
        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        
        if not isinstance(input_shape, tf.TensorShape):
            if isinstance(input_shape, (list, tuple)):
                input_shape = tf.TensorShape(input_shape)
            else:
                raise ValueError("Input shape must be a list, tuple, or TensorShape.")
        self.size_omega_part = self.filters * self.kernel_size * math.floor((self.kernel_size - 1) / 2)
        self.pad = math.floor((self.kernel_size - 1) / 2)
        

        # -----------------------------------------matrix_pad------------------------------------
        if self.padding == "SAME":
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
            raise ValueError("Padding not found. Use 'SAME' or 'VALID'.")

        # -------------------------------------------------------------------------------------

        # ------------------------------out_in_shape_phi_indices-------------------------------
        self.set_indices_phi()
        self.set_indices_ul_triangular()
        # -------------------------------------------------------------------------------------

        # \omega_diag
        self.omega_diag = tf.constant(np.random.uniform(size=(self.filters, self.kernel_size,), low=-0.05, high=0.05), dtype=tf.float32,name='omega_diag')

        # \omega_triu
        self.omega_triu = tf.constant(np.random.uniform(size=(1, self.size_omega_part), low=-0.05, high=0.05),
                                          dtype=tf.float32, name='omega_triu')

        # \omega_tril
        self.omega_tril = tf.constant(np.random.uniform(size=(1, self.size_omega_part), low=-0.05, high=0.05),
                                          dtype=tf.float32, name='omega_tril')

        # ---------------------------------indeice_Omega:part-----------------------------------
        self.set_indices_ul_triangular()
        # -------------------------------------------------------------------------------------

        # \use_lambda_in
        self.use_lambda_in = tf.random.uniform(shape=(self.kernel_size, 1), minval=-0.05, maxval=0.05,dtype=tf.float32, name='use_lambda_in')

        # \use_lambda_out
        self.use_lambda_out = tf.random.uniform(shape=(1, self.kernel_size), minval=-0.05, maxval=0.05,dtype=tf.float32, name='use_lambda_out')

        # use_encode
        if self.use_encode:
            self.use_encode =tf.random.uniform(shape=(1, self.Right_shape[0] * self.Right_shape[1]), minval=-0.05, maxval=0.05, dtype=tf.float32, name='use_encode')

        else:
            self.use_encode = tf.ones(shape=(1, self.Right_shape[0] * self.Right_shape[1]), dtype=tf.float32, name='use_encode')

        # use_decode
        if self.use_decode:
            self.use_decode = tf.random.uniform(shape=(self.output_lenght, 1), minval=-0.05, maxval=0.05,dtype=tf.float32, name='use_decode')      

        else:
            self.use_decode = tf.zeros(shape=(self.output_lenght, 1), dtype=tf.float32, name='use_decode')
        # ---------------------------------------------------------------------------------------
        self.custom = True
        self.base= self.get_base()
        self.kernel = self.get_kernel() 
        # ---------------------------------------------------------------------------------------
   
        # ---------------------------------------------------------------------------------------

    def conv(self,  inputs: tf.Tensor):
        if not self.custom:
            raise ValueError("The layer has not been built yet.")
        # ----------------------------------Inputs---------------------------------------------

        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        upFlatten = tf.matmul(a=self.matrix_pad, b=flatten)
        upFlatten=tf.transpose(upFlatten, perm=[2, 0, 1])
        
        
        # -----------------------------------Outputs-------------------------------------------------
        outputs = tf.matmul(a=upFlatten, b=self.kernel)
        outputs = tf.transpose(outputs, perm=[1, 2, 0])
        outputs = tf.reshape(outputs, shape=(-1, self.out_shape1, self.out_shape2, self.filters))

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass

        return outputs    
    
    @tf.function(jit_compile=True)
    def conv_jit(self, inputs: tf.Tensor):
         return self.conv(inputs)

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
    
    def get_omega(self, *args):
        if not self.custom:
            raise ValueError("The layer has not been built yet.")
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
        return omega

    def get_base(self, *args):
        if not self.custom:
            raise ValueError("The layer has not been built yet.")
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
    
    def get_kernel(self, *args):
        if not self.custom:
            raise ValueError("The layer has not been built yet.")
        # ------------------------------Build Noyau--------------------------------------------
        kernel = tf.reshape(self.base, shape=(self.filters, self.kernel_size * self.kernel_size))

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

        return kernel


if __name__ == '__main__':
    print("This is a module for simple convolutional operations. It contains functions and classes to handle convolution operations, including matrix representations and padding techniques.")