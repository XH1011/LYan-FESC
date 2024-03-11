import hdf5storage
import scipy.io as sio
import os, traceback
import pandas as pd

from AEnet_13_sc import ConvAE
from AEutils_sc import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VI+SIBLE_DEVICES"] = "0"

# d = 16
d = 10
# alpha = 6
alpha = 5
# ro = 0.03
ro = 0.008

# encoded_data_CWRU
data = hdf5storage.loadmat('./One-class-supervision/results/CWRU/DCAE/en_G_test_all.mat')
Img = data['data']
Label = data['gnd']
Img = np.reshape(Img, (Img.shape[0], 128))
order = list(range(len(Img)))

n_input = [128, 1]
kernel_size = [3]
n_hidden = [15]
batch_size = 7 * 300
model_path = './sub_clustering/results/CWRU/model.ckpt'
restore_path = './sub_clustering/results/CWRU/model.ckpt'
logs_path = './sub_clustering/results/CWRU/logs'

# # #encoded-glow(chopper)
# data = hdf5storage.loadmat('./One-class-supervision/results/Chopper/DCAE/en_G_test_all.mat')
# Img = data['data']
# Label = data['gnd']
# Img = np.reshape(Img, (Img.shape[0], 128))
# order = list(range(len(Img)))
#
# n_input = [128, 1]
# kernel_size = [3]
# n_hidden = [15]
# batch_size = 7 * 291
# model_path = './sub_clustering/results/Chopper/model.ckpt'
# restore_path = './sub_clustering/results/Chopper/model.ckpt'
# logs_path = './sub_clustering/results/Chopper/logs'

#CWRU
num_class = 7  # how many class we sample
num_sa = 300
# #Chopper
# num_class = 7
# num_sa = 291

batch_size_test = num_sa * num_class

iter_ft = 0
ft_times = 25
display_step = 300

fine_step = -1

reg1 = 1e-4

mm = 0
mreg = [0, 0, 0, 0]
mlr2 = 0
startfrom = [0, 0, 0]

def test_face(Img, Label, CAE, num_class,learning_rate):
    acc_ = []
    for i in range(0, 1):
        face_10_subjs = np.array(Img[num_sa * i:num_sa * (i + num_class), :])
        face_10_subjs = face_10_subjs.astype(float)
        label_10_subjs = np.array(Label[num_sa * i:num_sa * (i + num_class)])
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs)

        CAE.initlization()
        CAE.save_model()
        # CAE.restore()
        COLD = None
        lastr = 1.0
        losslist = []
        for iter_ft in range(ft_times):
            # print('iter_ft',iter_ft)
            CAE.restore()
            # cost, C, dd, dt = CAE.partial_fit(face_10_subjs, mode='fine')
            cost, C, dd, dt = CAE.partial_fit(face_10_subjs, learning_rate, mode='fine')  #
            CAE.save_model()
            losslist.append(cost[-1])
            if iter_ft % display_step == 0 and iter_ft > 10:
                print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
                print(cost)
                for posti in range(2):
                    display(C, face_10_subjs, d, alpha, ro, num_class, label_10_subjs)

            if COLD is not None:
                normc = np.linalg.norm(COLD, ord='fro')
                normcd = np.linalg.norm(C - COLD, ord='fro')
                r = normcd / normc
                # print(epoch,r)
                if r < 1.0e-6 and lastr < 1.0e-6:
                    print("early stop")
                    print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
                    print(cost)
                    for posti in range(2):
                        display(C, face_10_subjs, d, alpha, ro, num_class, label_10_subjs)
                    break
                lastr = r
            COLD = C

        # print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
        # print(cost)

        # drawC(C)
        # print(C)
        for posti in range(1):
            acc, L, y_pre,index_cluster, index_original,y_pre_o = display(C, face_10_subjs, d, alpha, ro, num_class, label_10_subjs)
            acc_.append(acc)
            probs = probas(L, y_pre, order, index_cluster, index_original)
            acc_probs = cal_acc(label_10_subjs, probs)
        acc_.append(acc)

    acc_ = np.array(acc_)
    # print(acc_)
    mm = np.max(acc_)

    # print("%d subjects:" % num_class)
    # print("Max: %.4f%%" % ((1 - mm) * 100))
    # print(acc_)
    lossnp = np.asarray(losslist)
    return (1 - mm),L,y_pre_o

all_subjects = [7]

for reg2 in [0.1,1,10,100]:
    for reg3 in [0.1,1,10,100]:
        for reg4 in [0.1,1,10,100]:
            for lr2 in [1e-4]:
                try:
                    print("reg:", reg2, reg3, reg4, lr2)
                    avg = []
                    med = []
                    iter_loop = 0
                    while iter_loop < len(all_subjects):
                        num_class = all_subjects[iter_loop]
                        batch_size = num_class * num_sa

                        tf.reset_default_graph()
                        CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2,
                                     re_constant3=reg3, re_constant4=reg4, ds=num_class, \
                                     kernel_size=kernel_size, batch_size=batch_size, model_path=model_path,
                                     restore_path=restore_path, logs_path=logs_path)
                        avg_i,L,y_pre_o = test_face(Img, Label, CAE, num_class, lr2)
                        avg.append(avg_i)
                        iter_loop = iter_loop + 1
                        visualize(Img, Label, CAE,'./sub_clustering/results/CWRU/clustering')
                        # visualize(Img, Label, CAE,'./sub_clustering/results/Chopper/clustering')
                    iter_loop = 0

                    if 1 - avg[0] > mm:
                        drawC(L, './sub_clustering/results/CWRU/L-CWRU-L2.png')
                        # drawC(L, './sub_clustering/CWRU/results/L-Chopper-L2.png')
                        mreg = [reg2, reg3, reg4, lr2]
                        mm = 1 - avg[0]
                    # print("max:", mreg, mm)

                except:
                    print("error in ", reg2, reg3, lr2)
                    # print("error in ", reg2, reg3)
                    traceback.print_exc()
                finally:
                    try:
                        CAE.sess.close()
                    except:
                        ''
