import pickle
import keras
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects


def LoadData_pickle(path, name, type='rb'):
    with open(path + name + '.pkl', type) as f:
        data = pickle.load(f)
    return data


def save_data(x, class_num, name):
    if name == 'train':
        file = open('./data/preprocessed/C' + class_num + '_train_pre' + '.pkl', 'wb')
        print(x.shape, 'Save data to the path:', './data/preprocessed/C' + class_num + '_train_pre' + '.pkl')
    else:
        file = open('./data/preprocessed/C' + class_num + '_test_pre' + '.pkl', 'wb')
        print(x.shape, 'Save data to the path:', './data/preprocessed/C' + class_num + '_test_pre' + '.pkl')
    pickle.dump(x, file)
    file.close()


# def leaky_relu(features, alpha=0.2):
#         f1 = 0.5 * (1 + alpha)
#         f2 = 0.5 * (1 - alpha)
#         return f1 * features + f2 * keras.backend.abs(features)
#         leaky_relu.__name__ = 'custom_activation'

class ReLUs(Activation):

    def __init__(self, activation, **kwargs):
        super(ReLUs, self).__init__(activation, **kwargs)
        self.__name__ = 'ReLU_s'


def leaky_relu(features, alpha=0.2):
    # Your activation function specialties here
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * features + f2 * keras.backend.abs(features)


get_custom_objects().update({'leaky_relu': ReLUs(leaky_relu)})
