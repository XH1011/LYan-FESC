import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, re_constant3=1.0,re_constant4=1.0,
                 batch_size=200, reg=None,ds = None, \
                 denoise=False, model_path=None, restore_path=None, \
                 logs_path='./logs',rawImg=None):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        usereg = 2
        # input required to be fed
        self.x = tf.placeholder(tf.float32, [None, n_input[0]* n_input[1]])
        self.learning_rate = tf.placeholder(tf.float32, [])
        t_bs = tf.shape(self.x)[0]
        weights = self._initialize_weights()

        if denoise == False:#不去噪
            x_input = self.x
            # latent, shape = self.encoder(x_input, weights)
        else:#去噪：添加随机噪声到原始数据x上
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))
            latent, shape = self.encoder(x_input, weights)


        # classifier  module
        if ds is not None: #ds:num class类别数
            pslb = tf.layers.dense(x_input,ds,kernel_initializer=tf.random_normal_initializer(),activation=tf.nn.softmax,name = 'ss_d')
            #创建全连接层  将输入z经过全连接层进行变换，得到一个具有ds维度的输出向量pslb
            cluster_assignment = tf.argmax(pslb, -1)
            eq = tf.to_float(tf.equal(cluster_assignment,tf.transpose(cluster_assignment)))#包含了每个样本所属类别的索引的向量
            # 计算两个样本是否属于同一个类别,不属于同一个类别，对应位置的元素为 1.0,否则为 0.0.  将cluster_assignment与转置后的cluster_assignment对比
        Coef = weights['Coef']
        self.Coef = Coef

        if usereg == 2:#正则化损失
            self.reg_losses = tf.reduce_sum(tf.square(self.Coef))+tf.trace(tf.square(self.Coef))
        else:
            self.reg_losses = tf.reduce_sum(tf.abs(self.Coef))+tf.trace(tf.abs(self.Coef))

        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        x_flattten = tf.reshape(x_input, [t_bs, -1]) #原始输入
        # x_flattten2 = tf.reshape(self.x_r, [t_bs, -1]) #重构输入 coef
        XZ = tf.matmul(Coef, x_flattten)
        self.selfexpress_losses2 = 0.5 * tf.reduce_sum(tf.square(tf.subtract(XZ, x_flattten)))

        normL = True
        #graph(C)
        absC = tf.abs(Coef)
        C = (absC + tf.transpose(
            absC)) * 0.5  # * (tf.ones([Coef.shape[0].value,Coef.shape[0].value])-tf.eye(Coef.shape[0].value))
        C = C + tf.eye(Coef.shape[0].value) #加上单位矩阵，确保矩阵 C 对角线上的元素为 1

        self.cc=C- tf.eye(Coef.shape[0].value) #矩阵 C 减去单位矩阵

        if normL == True:
            D = tf.diag(tf.sqrt((1.0 / tf.reduce_sum(C, axis=1))))
            I = tf.eye(D.shape[0].value)
            L = I - tf.matmul(tf.matmul(D, C), D)
            D = I
        else:
            D = tf.diag(tf.reduce_sum(C, axis=1))
            L = D - C


        self.d = cluster_assignment
        regass = tf.to_float(tf.reduce_sum(pslb,axis=0))

        onesl=np.ones(batch_size)
        zerosl=np.zeros(batch_size)
        #thershold
        weight_label=tf.where(tf.reduce_max(pslb,axis=1)>0.8,onesl,zerosl)
        cluster_assignment1=tf.one_hot(cluster_assignment,ds) #将其转换为one-hot编码表示 ds指定了one-hot的维度
        self.w_weight=weight_label
        self.labelloss=tf.losses.softmax_cross_entropy(onehot_labels=cluster_assignment1,logits=pslb,weights=weight_label)
        #softmax 交叉熵损失的值

        self.graphloss = tf.reduce_sum(tf.nn.relu((1-eq) * C)+tf.nn.relu(eq * (0.001-C)))+ tf.reduce_sum(tf.square(regass))
        #图损失的表达式


        self.loss3 = ( re_constant2 * self.selfexpress_losses2  + re_constant3 * self.labelloss+re_constant4 * self.graphloss)
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss3)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss3)
        self.optimizer = self.optimizer2#tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)  # GradientDescentOptimizer #AdamOptimizer
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if v.name.startswith("Coef")])
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        n_layers = len(self.n_hidden)
        all_weights['Coef'] = tf.Variable(
            1.0e-5 * (tf.ones([self.batch_size, self.batch_size], dtype=tf.float32)), name='Coef')
        return all_weights


    def partial_fit(self, X, lr, mode=0):  #
        cost0,   summary, _, Coef,d,dt = self.sess.run((self.selfexpress_losses2, self.merged_summary_op,
                                                        self.optimizer, self.Coef,self.w_weight,self.d),
                                                        feed_dict={self.x: X, self.learning_rate: lr})  #
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return [cost0], Coef, d,dt


    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)


    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        # print("model saved in file: %s" % save_path)

    def restore(self): #从指定的文件中恢复模型的参数和变量
        self.saver.restore(self.sess, self.restore_path)
        # print("model restored")
