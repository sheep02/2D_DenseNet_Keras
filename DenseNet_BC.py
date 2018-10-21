from keras.models import Model
from keras.layers import Input, Concatenate, concatenate, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation
import keras.backend as K


# Default values is for cifar100. 
class DenseNet():

    def __init__(self, shape_of_HW=(32,32), classes=100, eps=1e-7):
        self.eps = eps
        self.classes = classes

        if K.image_dim_ordering() == 'tf':
          self.concat_axis = 3
          self.img_input = Input(shape=shape_of_HW+(3,), name='data')
        else:
          self.concat_axis = 1
          self.img_input = Input(shape=(3,)+shape_of_HW, name='data')


    def conv(self, x, stage, branch, nb_filter, dropout_rate=None):

        conv_name_base = f"conv{stage}_{branch}"
        relu_name_base = f"relu{stage}_{branch}"
        
        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4  
        x = BatchNormalization(epsilon=self.eps, axis=self.concat_axis, name=f"{conv_name_base}_x1_bn")(x)
        x = Activation('relu', name=f"{relu_name_base}_x1")(x)
        x = Conv2D(filters=inter_channel, kernel_size=(1, 1), padding="same", use_bias=False, name=f"{conv_name_base}_x1")(x)
    
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
        # 3x3 Convolution
        x = BatchNormalization(epsilon=self.eps, axis=self.concat_axis, name=f"{conv_name_base}_x2_bn")(x)
        x = Activation('relu', name=f"{relu_name_base}_x2")(x)
        x = Conv2D(filters=nb_filter, kernel_size=(3, 3), padding="same", use_bias=False, name=f"{conv_name_base}_x2")(x)
    
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
        return x


    def dense_block(self, x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):

        concat_feat = x
    
        for i in range(nb_layers):
            branch = i+1
            x = self.conv(concat_feat, stage, branch, growth_rate, dropout_rate)
            concat_feat = concatenate([concat_feat, x], axis=self.concat_axis, name=f"concat_{stage}_{branch}")
    
            if grow_nb_filters:
                nb_filter += growth_rate
    
        return concat_feat, nb_filter   


    def transition_layer(self, x, stage, nb_filter, compression=1.0, dropout_rate=None):

        conv_name_base = f"conv{stage}_blk"
        relu_name_base = f"relu{stage}_blk"
        pool_name_base = f"pool{stage}" 
    
        x = BatchNormalization(epsilon=self.eps, axis=self.concat_axis, name=f"{conv_name_base}_bn")(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Conv2D(filters=int(nb_filter * compression), kernel_size=(1, 1), padding="same", use_bias=False, name=conv_name_base)(x)
    
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
        x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)
    
        return x  


    def build(self, nb_layers=None, depth=40, growth_rate=12, nb_dense_block=4, nb_filter=None, theata=0.5, dropout_rate=0.2, weights_path=None):

        nb_layers = [depth] * nb_dense_block

        if None == nb_filter:
            nb_filter = growth_rate * 2
    
        # Initial convolution
        x = Conv2D(filters=nb_filter, kernel_size=(7, 7), strides=(2, 2), padding="same", use_bias=False, name='conv1')(self.img_input)
        x = BatchNormalization(epsilon=self.eps, axis=self.concat_axis, name='conv1_bn')(x)
        x = Activation('relu', name='relu1')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    
        # Add dense blocks
        for block_idx in range(nb_dense_block):
            stage = block_idx + 2
            x, nb_filter = self.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate)

            # Add transition_layer
            x = self.transition_layer(x, stage, nb_filter, compression=theata, dropout_rate=dropout_rate)
            nb_filter = int(nb_filter * theata)
    
        stage = block_idx + 2
        x, nb_filter = self.dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate)
    
        x = BatchNormalization(epsilon=self.eps, axis=self.concat_axis, name=f"conv{stage}_blk_bn")(x)
        x = Activation('relu', name=f"relu{stage}_blk")(x)
        x = GlobalAveragePooling2D(name=f"pool{stage}")(x)
    
        x = Dense(self.classes, name='fc6')(x)
        x = Activation('softmax', name='prob')(x)
    
        model = Model(self.img_input, x, name='densenet')
    
        if weights_path is not None:
          model.load_weights(weights_path)
    
        return model