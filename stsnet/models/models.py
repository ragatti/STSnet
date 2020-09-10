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
    input1 = Input(shape=(ch_rows * ch_cols, samples))
    # -------------------------------------------------------------------------
    reshape1 = Reshape(target_shape=(1, ch_rows * ch_cols, samples))(input1)

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


def EEGnet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64,
           F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
    advised to do some model searching to get optimal performance on your
    particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of
    this parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D
    for overcomplete). We believe the main parameters to focus on are F1 and D.

    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    input_shape = (Chans, Samples)

    model = Sequential()
    model.add(Reshape(target_shape=(1,)+input_shape,
                      input_shape=input_shape))
    model.add(Conv2D(F1, (1, kernLength), padding='same', use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
              depthwise_constraint=max_norm(1.)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 4), data_format="channels_first"))
    model.add(dropoutType(dropoutRate))

    model.add(SeparableConv2D(F2, (1, 16), use_bias=False, padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 8), data_format="channels_first"))
    model.add(dropoutType(dropoutRate))

    model.add(Flatten(name='flatten'))

    model.add(Dense(nb_classes, name='dense',
                    kernel_constraint=max_norm(norm_rate)))
    model.add(Activation('softmax', name='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=["accuracy"])

    return model
