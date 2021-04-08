'''
SeaIce Conv 1D model
'''


import tensorflow as tf
from base_model import BaseModel


class SeaIceConv1D(BaseModel):

    def __init__(self, in_shape):
        super(SeaIceConv1D, self).__init__(in_shape)

    def makeModel(self, **kwargs):

        self.model = tf.keras.Sequential()
        nLayers = kwargs['nLayers']
        _filters = kwargs['filters']
        _kernel_size = kwargs['kernel_size']
        _strides = kwargs['strides'] if 'strides' in kwargs else [1] * nLayers
        _paddings = kwargs['paddings']
        _activation = kwargs['activations']
        _useBias = kwargs['use_bias']
        _kernelInitializer = kwargs['kernel_initializer'] if 'kernel_initializer' in kwargs else 'glorot_uniform'
        _biasInitializer = kwargs['bias_initializer'] if 'bias_initializer' in kwargs else 'zeros'
        _batchNormalize = kwargs['batch_normalize']
        _outputActivation = kwargs['output_activation'] if 'output_activation' in kwargs else None

        self.model.add(tf.keras.Input(shape = (self.in_shape), name="input"))

        for i in range(nLayers):
            self.model.add(tf.keras.layers.Conv1D(
                filters=_filters[i], kernel_size=_kernel_size[i], strides=_strides[i], padding=_paddings[i], use_bias=_useBias[i], avtivation=_activation[i], kernel_initializer=_kernelInitializer[i], bias_initializer=_biasInitializer[i]), name="Conv1D_{}".format(i))
            # batch normalization layers
            if _batchNormalize[i]:
                self.model.add(tf.keras.layers.BatchNormalization(), name="Batch_Norm_{}".format(i))

        if _outputAcivation is not None:
            self.model.add(tf.keras.layers.Activation(_outputActivation, name="output"))