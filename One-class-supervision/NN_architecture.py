from keras.layers import *
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from util import leaky_relu, ReLUs

def build_basic_model(in_channel):  #(build_basic_model(3*2**(i+1))) in_channel=6,12,24 (i=0,1,2)
    ##in_channel=2,4,8 (i=0,1,2)
    """基础模型，即耦合层中的模型（basic model for Coupling）
    """
    _in = Input(shape=(None, None, in_channel)) ##（？，？，？，2）
    _ = _in
    hidden_dim = 512
    # get_custom_objects().update({'leaky_relu': Activation(leaky_relu)})
    get_custom_objects().update({'leaky_relu': ReLUs(leaky_relu)})
    _ = Conv2D(hidden_dim,      #卷积核个数,卷积输出的通道数（最后一位）
               (3, 3),          #卷积核尺寸
               padding='same')(_)    #当padding=same时，如果stride为1，输入和输出的尺寸是一样的
    ##（？，？，？，512）
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    _ = Activation('leaky_relu')(_)
    _ = Conv2D(hidden_dim,
               (1, 1),
               padding='same')(_)
    # _ = Actnorm(add_logdet_to_loss=False)(_)
    ##（？，？，？，512）
    _ = Activation('leaky_relu')(_)
    _ = Conv2D(in_channel,
               (3, 3),
               kernel_initializer='zeros',  #最后一层使用零初始化，这样就使得初始状态下输入输出一样，即初始状态为一个恒等变换，这有利于训练深层网络
               padding='same')(_)
    ##（？，？，？，2）
    return Model(_in, _)  #(768, 1, 2, 6)



