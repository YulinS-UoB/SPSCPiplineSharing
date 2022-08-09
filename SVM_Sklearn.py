# ***********************************
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from skimage import filters
import pickle
import time
import datetime
import cv2 as cv
import os
# ***********************************
DataRoot = 'data/AnnotationDS2022-08-04_17_32/'
DataBar = 5


def generateDSet(data_bar, exclude=None, data_root=DataRoot, seeding='Auto'):
    # seeding can either be 'Auto' as default or a number which makes random dataset reproducable
    # format of exclude: (seed, exclude size)
    if seeding == 'Auto':
        seed1 = np.random.randint(3000)
        print('The seed for reading file is decided as: ', seed1)
        seed2 = np.random.randint(3000)
        print('The seed for shuffling data is decided as: ', seed2)
    else:
        seed1 = seeding[0]
        seed2 = seeding[1]
    neg_data_list = os.listdir(data_root + '001/')
    neg_data_list.sort()
    pos_data_list = os.listdir(data_root + '002/')
    pos_data_list.sort()
    neg_data_num = len(neg_data_list)
    pos_data_num = len(pos_data_list)
    if type(data_bar) == int:
        if exclude:
            np.random.seed(exclude[0])
            niet_neg = np.random.choice(list(range(neg_data_num)), size=exclude[1])
            niet_pos = np.random.choice(list(range(pos_data_num)), size=exclude[1])
            list_left_neg = np.setdiff1d(list(range(neg_data_num)), niet_neg)
            list_left_pos = np.setdiff1d(list(range(pos_data_num)), niet_neg)
        else:
            list_left_neg = np.asarray(range(neg_data_num))
            list_left_pos = np.asarray(range(pos_data_num))
        np.random.seed(seed1)
        itrtr_neg = np.random.choice(list_left_neg, size=data_bar)
        itrtr_pos = np.random.choice(list_left_pos, size=data_bar)
        itrtr = range(data_bar)
    elif type(data_bar) == list:
        print('Warning: no random choice of data file was performied as an ordered list was given')
        itrtr = range(len(data_bar))
        itrtr_neg = data_bar[0:(len(data_bar) // 2)]
        itrtr_pos = data_bar[0:(len(data_bar) // 2)]
    else:
        print('Error in data_bar type!')
        return 1
    data_list = []
    label_list = []
    for i in itrtr:
        data_list.append(np.load("{}001/{}" + neg_data_list[itrtr_neg[i]]))
        label_list.append(np.ones((data_list[0].shape[0])))
        data_list.append(np.load("{}002/{}" + pos_data_list[itrtr_pos[i]]))
        label_list.append(2 * np.ones((data_list[0].shape[0])))
    test_data = np.concatenate(data_list).astype('int')
    test_label = np.concatenate(label_list).astype('int')
    np.random.seed(seed2)
    np.random.shuffle(test_data)
    np.random.seed(seed2)
    np.random.shuffle(test_label)

    return test_data, test_label


x, y = generateDSet(data_bar=[0, 1, 2, 3, 4])

model_tf = tf.keras.Sequential([
    tf.keras.Input(shape=(48,), batch_size=800),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),

])
model_tf.compile(
    optimizer='adam',
    loss='hinge',
    metrics=['categorical_accuracy']
)

clf3 = MLPClassifier(batch_size=800, activation='logistic', hidden_layer_sizes=(24, 12, 4, 1), random_state=1428,
                     verbose=True, n_iter_no_change=1500, max_iter=1500)


x2, y2 = generateDSet(data_bar='T00000B001.npy')
x3, y3 = generateDSet(data_bar='T00000B002.npy')
x4, y4 = generateDSet(data_bar='T00000B003.npy')
x5, y5 = generateDSet(data_bar='T00000B004.npy')
clf3.fit(x, y)
clf3.fit(x2, y2)
clf3.fit(x3, y3)
clf3.fit(x4, y4)
y5H = clf3.predict(x5)

testImg = np.load('data/AnnotationDS2022-08-04_17_32/Processed_Stack_T00000.npy')
testX = testImg[:, :, :, 0].reshape((48, 2160 * 2560))
testX = np.moveaxis(testX, 1, 0)
t1 = time.time()
testY = clf3.predict(testX)
print('Time consumed: {} secs'.format(time.time()-t1))
testYNDA = testY.reshape((2160, 2560))
plt.imshow(testYNDA)
plt.colorbar()

testYNDA_Blurred = filters.gaussian(testYNDA, sigma=3, preserve_range=True)
testYNDA_Blurred[testYNDA_Blurred < 1.25] = 0
testYNDA_Blurred[testYNDA_Blurred > 0] = 1
plt.imshow(testYNDA_Blurred)
plt.colorbar()

with open('{}model_{}.pkl'.format(DataRoot, datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')), 'wb') as pkf:
    pickle.dump(clf3, pkf)

cv.imwrite('{}FinalImgPrdct.tiff'.format(DataRoot), (testYNDA_Blurred * 32767).astype('uint16'))

