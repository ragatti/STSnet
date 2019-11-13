"""STSnet model
"""

from __future__ import division, print_function, absolute_import
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Reshape
from tensorflow.keras.layers import SpatialDropout2D, TimeDistributed
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Flatten, Permute, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import numpy as np

if K.image_data_format() == 'channels_last':
    K.set_image_data_format("channels_first")
    print('Set data format to', K.image_data_format())


def STSnet(n_clss, ch_rows=4, ch_cols=5, samples=128, kernLength=None, F1=4,
           D=2, F2=8, norm_rate=0.25, dropoutRate=0.25, dropoutType='Dropout'):
    """Keras Implementation of STSnet.

    # Arguments:
   
        n_clss: int, number of classes to clasify (default 4)
        ch_rows: number of rows of 2D spatial distribution (default 4)
        ch_cols: number of columns of 2D spatial distribution (default 5)
        samples: number of epoch samples form each trial (default 128)
        dropoutRate: dropout value (default 0.25)
        kernLength: kernel length value in Conv2D layer(default None)
        F1: number of filters in Conv2D layer (default 4)
        D: deep multiplier to DepthwiseConv2D (default 2)
        F2: number of filters to each kernel from SeparableConv2D (default 8)
        norm_rate: maximum norm for the incoming dense weights(default 0.25)
        dropoutType: 'SpatialDropout2D' or 'Dropout' (defautl 'Dropout')
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    if kernLength is None:
        kernLength = np.int(samples/2)
    print('kernel_length', kernLength)
    # -------------------------------------------------------------------------
    # DEPTWISE 2D MODEL
    DW_in = Input(shape=(F1, ch_rows, ch_cols))
    DW_out = DepthwiseConv2D((ch_rows, ch_cols), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(DW_in)
    DW = Model(inputs=DW_in, outputs=DW_out)
    # -------------------------------------------------------------------------
    input1 = Input(shape=(ch_rows, ch_cols, samples))
    # -------------------------------------------------------------------------
    reshape1 = Reshape(target_shape=(1, ch_rows*ch_cols, samples))(input1)

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, ch_rows*ch_cols, samples),
                    use_bias=False)(reshape1)
    block1 = BatchNormalization(axis=1)(block1)
    reshape2 = Reshape(target_shape=(F1, ch_rows, ch_cols, samples))(block1)
    perm_dims1 = 4, 1, 2, 3
    perm1 = Permute(perm_dims1)(reshape2)
    block1 = TimeDistributed(DW)(perm1)
    perm_dims2 = 2, 3, 4, 1
    perm2 = Permute(perm_dims2)(block1)
    reshape3 = Reshape(target_shape=(F1*D, 1, samples))(perm2)
    block1 = BatchNormalization(axis=1)(reshape3)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, np.int(kernLength/16)))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, np.int(kernLength/8)))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(n_clss, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)
