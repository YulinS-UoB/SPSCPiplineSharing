# ***********************************
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
DataRoot = 'data/AnnotationDS2022-08-09_21_39/'


def generateDSet(data_bar, data_root=DataRoot, exclude=None, seeding=None, batch_size=800):
    # seeding can either be 'Auto' as default or a number which makes random dataset reproducable
    # format of exclude: (seed, exclude size)
    if seeding is None:
        seed1 = np.random.randint(3000)
        print('The seed for reading file is decided as: ', seed1)
        seed2 = np.random.randint(3000)
        print('The seed for shuffling data is decided as: ', seed2)
    else:
        seed1 = seeding[0]
        print('The seed for reading file is input as: ', seed1)
        seed2 = seeding[1]
        print('The seed for shuffling data is input as: ', seed2)
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
            list_left_pos = np.setdiff1d(list(range(pos_data_num)), niet_pos)
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
        data_list.append(np.load("{}001/{}".format(data_root, neg_data_list[itrtr_neg[i]])))
        label_list.append(np.ones((data_list[-1].shape[0])))
        data_list.append(np.load("{}002/{}".format(data_root, pos_data_list[itrtr_pos[i]])))
        label_list.append(2 * np.ones((data_list[-1].shape[0])))
    test_data = np.concatenate(data_list).astype('int')
    test_label = np.concatenate(label_list).astype('int')
    np.random.seed(seed2)
    np.random.shuffle(test_data)
    np.random.seed(seed2)
    np.random.shuffle(test_label)
    test_data = test_data[0:test_data.shape[0] // (batch_size * 10) * (batch_size * 10), :]
    test_label = test_label[0:test_label.shape[0] // (batch_size * 10) * (batch_size * 10)] - 1

    return test_data, test_label


x_test, y_test = generateDSet(data_bar=5, seeding=(1428, 2500))

x_val, y_val = generateDSet(data_bar=2, exclude=(1428, 5))


model_tf = tf.keras.Sequential([
    tf.keras.Input(shape=(48,), batch_size=800),
    tf.keras.layers.Dense(units=24, activation='sigmoid'),
    tf.keras.layers.Dense(units=12, activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),

])
model_tf.compile(
    optimizer='adam',
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)


model_tf.fit(x_test, y_test, batch_size=800, epochs=100)


clf2 = RandomForestRegressor(n_estimators=1000, n_jobs=5, verbose=1)

clf3 = MLPClassifier(batch_size=800, activation='logistic', hidden_layer_sizes=(48, 24, 12, 6, 1), random_state=1428,
                     verbose=True, n_iter_no_change=50, max_iter=50)
clf2.fit(x_test, y_test)
clf3.fit(x_test, y_test)

y_valH2 = clf2.predict(x_val)
y_valH = clf3.predict(x_val)

testImg = np.load('data/AnnotationDS2022-08-04_17_32/Processed_Stack_T00000.npy')
testX = testImg[:, :, :, 0].reshape((48, 2160 * 2560))
testX = np.moveaxis(testX, 1, 0)
t1 = time.time()
testY = clf3.predict(testX)
print('Time consumed: {} secs'.format(time.time()-t1))
testYNDA = testY.reshape((2160, 2560))
plt.imshow(testYNDA)
plt.colorbar()

t1 = time.time()
testY2 = clf2.predict(testX)
print('Time consumed: {} secs'.format(time.time()-t1))
testYNDA2 = testY2.reshape((2160, 2560))
plt.imshow(testYNDA2)
plt.colorbar()

t1 = time.time()
testY2 = model_tf.predict(testX, batch_size=800)
print('Time consumed: {} secs'.format(time.time()-t1))
testYNDA2 = testY2.reshape((2160, 2560))
plt.imshow(testYNDA2)
plt.colorbar()

testYNDA_Blurred = filters.gaussian(testYNDA, sigma=2, preserve_range=True)
testYNDA_Blurred[testYNDA_Blurred < 0.5] = 0
testYNDA_Blurred[testYNDA_Blurred > 0] = 1
plt.imshow(testYNDA_Blurred)
plt.colorbar()

with open('{}model_{}.pkl'.format(DataRoot, datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')), 'wb') as pkf:
    pickle.dump(clf3, pkf)

cv.imwrite('{}FinalImgPrdct.tiff'.format(DataRoot), (testYNDA_Blurred * 32767).astype('uint16'))
