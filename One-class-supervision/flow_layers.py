#! -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import keras.backend as K
from keras.layers import Layer
import tensorflow as tf

class Permute(Layer):
    """排列层，提供两种方式重新排列最后一个维度的数据
    一种是直接反转，一种是随机打乱，默认是直接反转维度
    New Permute layer. Reverse or shuffle the final axis of inputs
    """
    def __init__(self, mode='reverse', **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.idxs = None # 打乱顺序的序id
        self.mode = mode
    def build(self, input_shape):
        super(Permute, self).build(input_shape)   #[768,1,4]
        in_dim = input_shape[-1]
        if self.idxs is None:
            if self.mode == 'reverse':
                self.idxs = self.add_weight(name='idxs',
                                            shape=(input_shape[-1],),
                                            dtype='int32',
                                            initializer=self.reverse_initializer,
                                            trainable=False)
            elif self.mode == 'random':
                self.idxs = self.add_weight(name='idxs',
                                            shape=(input_shape[-1],),
                                            dtype='int32',
                                            initializer=self.random_initializer,
                                            trainable=False)
    def reverse_initializer(self, shape, dtype=None):
        idxs = list(range(shape[0]))
        return idxs[::-1]  #最后一个元素到第一个元素
    def random_initializer(self, shape, dtype=None):
        # idxs = range(shape[0])
        idxs = list(range(shape[0]))
        np.random.shuffle(idxs)
        return idxs
    def call(self, inputs):
        num_axis = K.ndim(inputs)  #整数形式返回张量中的轴数
        inputs = K.permute_dimensions(inputs, list(range(num_axis)[::-1]))  #重新排列张量的轴
        x_outs = K.gather(inputs, self.idxs)   #在张量b中检索索引a的元素，返回与a类型相同的张量
        x_outs = K.permute_dimensions(x_outs, list(range(num_axis)[::-1]))
        return x_outs
    def inverse(self):
        in_dim = K.int_shape(self.idxs)[0]
        reverse_idxs = tf.nn.top_k(self.idxs, in_dim)[1][::-1]
        layer = Permute()
        layer.idxs = reverse_idxs
        return layer
    def get_config(self):
        config = {'mode': self.mode}
        base_config = super(Permute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InvDense(Layer):
    """可逆全连接层（对最后一个维度做共享权重的全连接）
    如果用在图像中，相当于1x1的卷积。
    Invertible dense layer of final axis.
    If inputs is image, it equals 1x1 invertible Conv2D.
    """
    def __init__(self,
                 isinverse=False,
                 **kwargs):
        super(InvDense, self).__init__(**kwargs)
        self.kernel = None
        self.isinverse = isinverse
    def initializer(self, shape):
        """通过随机正交矩阵进行LU分解初始化
        """
        import scipy as sp
        import scipy.linalg
        random_matrix = sp.random.randn(shape[-1], shape[-1])
        random_orthogonal = sp.linalg.qr(random_matrix)[0]
        p, l, u = sp.linalg.lu(random_orthogonal)
        u_diag_sign = sp.sign(sp.diag(u))
        u_diag_abs_log = sp.log(abs(sp.diag(u)))
        l_mask = 1 - sp.tri(shape[-1]).T # l的mask，下三角全1阵（但对角线全0）
        u_mask = 1 - sp.tri(shape[-1]) # u的mask，上三角全1阵（但对角线全0）
        # random_matrix = np.random.randn(shape[-1], shape[-1])
        # random_orthogonal = sp.linalg.qr(random_matrix)[0]
        # p, l, u = sp.linalg.lu(random_orthogonal)
        # u_diag_sign = np.sign(np.diag(u))
        # u_diag_abs_log = np.log(abs(np.diag(u)))
        # l_mask = 1 - np.tri(shape[-1]).T # l的mask，下三角全1阵（但对角线全0）
        # u_mask = 1 - np.tri(shape[-1]) # u的mask，上三角全1阵（但对角线全0）
        return p, l, u, u_diag_sign, u_diag_abs_log, l_mask, u_mask
    def build(self, input_shape):
        super(InvDense, self).build(input_shape)
        if self.kernel is None:
            (p, l, u, u_diag_sign, u_diag_abs_log,
                l_mask, u_mask) = self.initializer(input_shape)
            self.kernel_p = self.add_weight(name='kernel_p',
                                            shape=p.shape,
                                            initializer=lambda _: p,
                                            trainable=False)
            self.kernel_l = self.add_weight(name='kernel_l',
                                            shape=l.shape,
                                            initializer=lambda _: l,
                                            trainable=True)
            self.kernel_u = self.add_weight(name='kernel_u',
                                            shape=u.shape,
                                            initializer=lambda _: u,
                                            trainable=True)
            self.kernel_u_diag_sign = self.add_weight(name='kernel_u_diag_sign',
                                                         shape=u_diag_sign.shape,
                                                         initializer=lambda _: u_diag_sign,
                                                         trainable=False)
            self.kernel_u_diag_abs_log = self.add_weight(name='kernel_u_diag_abs_log',
                                                         shape=u_diag_abs_log.shape,
                                                         initializer=lambda _: u_diag_abs_log,
                                                         trainable=True)
            self.kernel_l = self.kernel_l * l_mask + K.eye(input_shape[-1])
            self.kernel_u = self.kernel_u * u_mask + tf.diag(
                self.kernel_u_diag_sign * K.exp(self.kernel_u_diag_abs_log))
            self.kernel = K.dot(K.dot(self.kernel_p, self.kernel_l),
                                self.kernel_u)
    def call(self, inputs):
        if self.isinverse:
            logdet = K.sum(self.kernel_u_diag_abs_log)
            x_outs = K.dot(inputs, tf.matrix_inverse(self.kernel))
        else:
            logdet = -K.sum(self.kernel_u_diag_abs_log)
            x_outs = K.dot(inputs, self.kernel)
        if K.ndim(inputs) > 2:
            logdet *= K.prod(K.cast(K.shape(inputs)[1:-1], 'float32'))
        self.logdet = logdet
        return x_outs
    def inverse(self):
        layer = InvDense(not self.isinverse)
        layer.kernel = self.kernel
        layer.kernel_u_diag_abs_log = self.kernel_u_diag_abs_log
        return layer
    def get_config(self):
        config = {'isinverse': self.isinverse}
        base_config = super(InvDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Split(Layer):
    """将输入分区沿着最后一个轴为切分为若干部分
    pattern：切分模式，记录每一部分的大小的list；默认对半切分为两部分
    split inputs into several parts according pattern
    """
    def __init__(self, pattern=None, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.pattern = pattern
    def call(self, inputs):
        if self.pattern is None:
            in_dim = K.int_shape(inputs)[-1]   #4 ori=12
            self.pattern = [in_dim//2, in_dim - in_dim//2]    #[2,2] ori=[6,6]
        partion = [0] + list(np.cumsum(self.pattern)) #在某一指定轴计算张量中的值的累加和,返回由中间结果组成的数组 [0,2,4]  ori=[0,6,12]
        return [inputs[..., i:j] for i,j in zip(partion, partion[1:])]   #[1:]：去掉第一个元素
    def compute_output_shape(self, input_shape):  #[(768, 1, 2), (768, 1, 2)]
        return [input_shape[:-1] + (d,) for d in self.pattern]
    def inverse(self):
        layer = Concat()
        return layer
    def get_config(self):
        config = {'pattern': self.pattern}
        base_config = super(Split, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Concat(Layer):
    """把最后一个轴拼接起来
    like Concatenate but add inverse()
    """
    def __init__(self, **kwargs):
        super(Concat, self).__init__(**kwargs)
    def call(self, inputs):
        self.pattern = [K.int_shape(i)[-1] for i in inputs]
        return K.concatenate(inputs, -1)
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (sum(self.pattern),)
    def inverse(self):
        layer = Split(self.pattern)
        return layer




class AffineCouple(Layer):  #layer([x1, x2, shift]) shift = self.shift_model(x1)
    """仿射耦合层
    """
    def __init__(self,
                 isinverse=False,
                 **kwargs):
        super(AffineCouple, self).__init__(**kwargs)
        self.isinverse = isinverse
    def call(self, inputs):
        """如果inputs的长度为3，那么就是加性耦合，否则就是一般的仿射耦合。
        if len(inputs) == 3, it equals additive coupling.
        if len(inputs) == 4, it is common affine coupling.
        """
        if len(inputs) == 3:
            x1, x2, shift = inputs
            log_scale = K.constant([0.])   #创建一个常数张量
        elif len(inputs) == 4:
            x1, x2, shift, log_scale = inputs
        if self.isinverse:
            logdet = K.sum(K.mean(log_scale, 0)) # 对数行列式
            x_outs = [x1, K.exp(-log_scale) * (x2 - shift)]
        else:
            logdet = -K.sum(K.mean(log_scale, 0)) # 对数行列式
            x_outs = [x1, K.exp(log_scale) * x2 + shift]
        self.logdet = logdet
        return x_outs
    def inverse(self):
        layer = AffineCouple(not self.isinverse)
        return layer
    def get_config(self):
        config = {'isinverse': self.isinverse}
        base_config = super(AffineCouple, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoupleWrapper:   #CoupleWrapper(build_basic_model(3*2**(i+1)))
    """仿射耦合层的封装，使得可以直接将模型作为参数传入
    just a wrapper of AffineCouple for simpler use.
    """
    def __init__(self,
                 shift_model,
                 log_scale_model=None,
                 isinverse=False):
        self.shift_model = shift_model
        self.log_scale_model = log_scale_model
        self.layer = AffineCouple(isinverse)
    def __call__(self, inputs, whocare=0):
        x1, x2 = inputs
        shift = self.shift_model(x1)
        if whocare == 0:
            layer = self.layer
        else:
            layer = self.layer.inverse()
        if self.log_scale_model is None:
            return layer([x1, x2, shift])  # √
        else:
            log_scale = self.log_scale_model(x1)
            return layer([x1, x2, shift, log_scale])
    def inverse(self):
        return lambda inputs: self(inputs, 1)


class Actnorm(Layer):
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self,
                 isinverse=False,
                 use_shift=True,
                 **kwargs):
        super(Actnorm, self).__init__(**kwargs)
        self.log_scale = None
        self.shift = None
        self.isinverse = isinverse
        self.use_shift = use_shift
    def build(self, input_shape):    # 为该层创建一个可训练的权重 根据编写你自己的层
        super(Actnorm, self).build(input_shape)
        kernel_shape = (1,)*(len(input_shape)-1) + (input_shape[-1],)  # [32,32,12]->[1,1,12] [768,1,4]->[1,1,4]
        ##(1,1,1,4)
        if self.log_scale is None:
            self.log_scale = self.add_weight(name='log_scale',
                                             shape=kernel_shape,
                                             initializer='zeros',
                                             trainable=True)
        if self.use_shift and self.shift is None:
            self.shift = self.add_weight(name='shift',
                                         shape=kernel_shape,
                                         initializer='zeros',
                                         trainable=True)
        if not self.use_shift:
            self.shift = 0.
    def call(self, inputs):
        if self.isinverse:
            logdet = K.sum(self.log_scale)
            x_outs = K.exp(-self.log_scale) * (inputs - self.shift)  #指数运算
        else:
            logdet = -K.sum(self.log_scale)
            x_outs = K.exp(self.log_scale) * inputs + self.shift ##先将log_scale进行指数运算，然后在与inputs对应的值相乘，最后加上shift(对每个元素进行平移操作)
        if K.ndim(inputs) > 2:   #张量中的轴数（维数）
            # a = K.shape(inputs)[1:-1]
            # b = K.cast(K.shape(inputs)[1:-1], 'float32')
            c = K.prod(K.cast(K.shape(inputs)[1:-1], 'float32'))
            logdet *= K.prod(K.cast(K.shape(inputs)[1:-1], 'float32')) #*=;logdet*将shape去头尾转换为float32的值的乘积；
        self.logdet = logdet
        return x_outs
    def inverse(self):
        layer = Actnorm(not self.isinverse)
        layer.log_scale = self.log_scale
        layer.shift = self.shift
        return layer
    def get_config(self):
        config = {'isinverse': self.isinverse, 'use_shift': self.use_shift}
        base_config = super(Actnorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class CondActnorm(Layer):   ##[(768, 1, 2), (768, 1, 2)]
    """双输入缩放平移变换层（Conditional scale and shift）
    将x1做缩放平移，其中缩放平移量由x2算出来
    返回变换后的x1
    """
    def __init__(self,
                 isinverse=False,
                 use_shift=True,
                 **kwargs):
        super(CondActnorm, self).__init__(**kwargs)  # 对继承自父类的属性进行初始化,子类把父类的__init__()放到自己的当中
        self.kernel = None
        self.bias = None
        self.isinverse = isinverse
        self.use_shift = use_shift
    def build(self, input_shape):   #build函数仅在Call被第一次调用时执行
        super(CondActnorm, self).build(input_shape)
        in_dim = input_shape[0][-1]
        if self.use_shift:
            out_dim = in_dim * 2
        else:
            out_dim = in_dim
        if self.kernel is None:
            self.kernel = self.add_weight(name='kernel',
                                          shape=(3, 3, in_dim, out_dim),
                                          initializer='zeros',
                                          trainable=True)
        if self.bias is None:
            self.bias = self.add_weight(name='bias',
                                        shape=(out_dim,),
                                        initializer='zeros',
                                        trainable=True)
    def call(self, inputs):
        x1, x2 = inputs
        in_dim = K.int_shape(x1)[-1]  #2
        x2_conv2d = K.conv2d(x2, self.kernel, padding='same')
        x2_conv2d = K.bias_add(x2_conv2d, self.bias)  #给张量添加一个偏置向量 #向量与矩阵的每一行相加，结果和矩阵大小相同
        if self.use_shift:
            log_scale,shift = x2_conv2d[..., :in_dim], x2_conv2d[..., in_dim:]  #索引最后一个维度的元素 含头不含尾
        else:
            log_scale,shift = x2_conv2d, 0.
        if self.isinverse:
            logdet = K.sum(K.mean(log_scale, 0))
            x_outs = K.exp(-log_scale) * (x1 - shift)
        else:
            logdet = -K.sum(K.mean(log_scale, 0))
            x_outs = K.exp(log_scale) * x1 + shift
        self.logdet = logdet
        return x_outs
    def inverse(self):
        layer = CondActnorm(not self.isinverse)
        layer.kernel = self.kernel
        layer.bias = self.bias
        return layer
    def get_config(self):
        config = {'isinverse': self.isinverse, 'use_shift': self.use_shift}
        base_config = super(CondActnorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Reshape(Layer):
    """重新定义Reshape层，默认为Flatten
    主要目的是添加inverse方法
    combination of keras's Reshape and Flatten. And add inverse().
    """
    def __init__(self, shape=None, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.shape = shape
    def call(self, inputs):
        self.in_shape = [i or -1 for i in K.int_shape(inputs)]
        if self.shape is None:
            self.shape = [-1, np.prod(self.in_shape[1:])]
        return K.reshape(inputs, self.shape)
    def compute_output_shape(self, input_shape):
        return tuple([i if i != -1 else None for i in self.shape])
    def inverse(self):
        return Reshape(self.in_shape)
    def get_config(self):
        config = {'shape': self.shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Squeeze(Layer):
    """shape=[h, w, c] ==> shape=[h/n, w/n, n*n*c]
    """
    def __init__(self, factor=2, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.factor = factor
    def call(self, inputs):
        height, width, channel = K.int_shape(inputs)[1:]  #1，1024，1
        # print('K.int_shape(inputs)',K.int_shape(inputs))
        assert width % self.factor == 0  #如果为真，程序继续执行；如果为假，则引发 AssertionError 异常，停止程序的执行
        # assert height % self.factor == 0 and width % self.factor == 0
        inputs = K.reshape(inputs, (-1,
                                    height,
                                    self.factor,
                                    width//(self.factor**2),
                                    self.factor,
                                    channel)) #(none,1,2,256,2,1)
        inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5)) #(none,1,256,2,2,1)  K.permute_dimensions 用于重新排列张量的维度
        x_outs = K.reshape(inputs, (-1,
                                     height,
                                     width//(self.factor**2),
                                     channel*self.factor**2)) #(none,1,256,4)
        return x_outs
    def compute_output_shape(self, input_shape):
        height, width, channel = input_shape[1:]
        return  (None, height,
                 width//(self.factor**2), channel*self.factor**2)
    def inverse(self):
        layer = UnSqueeze(self.factor)
        return layer
    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(Squeeze, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# class Squeeze(Layer):
#     """shape=[h, w, c] ==> shape=[h/n, w/n, n*n*c]
#     """
#     def __init__(self, factor=2, **kwargs):
#         super(Squeeze, self).__init__(**kwargs)
#         self.factor = factor
#     def call(self, inputs):
#         height, width, channel = K.int_shape(inputs)[1:]
#         assert height % self.factor == 0 and width % self.factor == 0
#         inputs = K.reshape(inputs, (-1,
#                                     height//self.factor,
#                                     self.factor,
#                                     width//self.factor,
#                                     self.factor,
#                                     channel))
#         inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5))
#         x_outs = K.reshape(inputs, (-1,
#                                      height//self.factor,
#                                      width//self.factor,
#                                      channel*self.factor**2))
#         return x_outs
#     def compute_output_shape(self, input_shape):
#         height, width, channel = input_shape[1:]
#         return  (None, height//self.factor,
#                  width//self.factor, channel*self.factor**2)
#     def inverse(self):
#         layer = UnSqueeze(self.factor)
#         return layer


class UnSqueeze(Layer):
    """shape=[h, w, c] ==> shape=[h*n, w*n, c/(n*n)]
    """
    def __init__(self, factor=2, **kwargs):
        super(UnSqueeze, self).__init__(**kwargs)
        self.factor = factor
    def call(self, inputs):
        height, width, channel = K.int_shape(inputs)[1:]
        assert channel % (self.factor**2) == 0
        inputs = K.reshape(inputs, (-1,
                                    height,
                                    width,
                                    self.factor,
                                    self.factor,
                                    channel//(self.factor**2)))
        inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5))
        x_outs = K.reshape(inputs, (-1,
                                     height,
                                     width*self.factor**2,
                                     channel//(self.factor**2)))
        return x_outs
    def compute_output_shape(self, input_shape):
        height, width, channel = input_shape[1:]
        return  (None, height,
                 width*self.factor**2, channel//(self.factor**2))
    def inverse(self):
        layer = Squeeze(self.factor)
        return layer
    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(UnSqueeze, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# class UnSqueeze(Layer):
#     """shape=[h, w, c] ==> shape=[h*n, w*n, c/(n*n)]
#     """
#     def __init__(self, factor=2, **kwargs):
#         super(UnSqueeze, self).__init__(**kwargs)
#         self.factor = factor
#     def call(self, inputs):
#         height, width, channel = K.int_shape(inputs)[1:]
#         assert channel % (self.factor**2) == 0
#         inputs = K.reshape(inputs, (-1,
#                                     height,
#                                     width,
#                                     self.factor,
#                                     self.factor,
#                                     channel//(self.factor**2)))
#         inputs = K.permute_dimensions(inputs, (0, 1, 3, 2, 4, 5))
#         x_outs = K.reshape(inputs, (-1,
#                                      height*self.factor,
#                                      width*self.factor,
#                                      channel//(self.factor**2)))
#         return x_outs
#     def compute_output_shape(self, input_shape):
#         height, width, channel = input_shape[1:]
#         return  (None, height*self.factor,
#                  width*self.factor, channel//(self.factor**2))
#     def inverse(self):
#         layer = Squeeze(self.factor)
#         return layer
