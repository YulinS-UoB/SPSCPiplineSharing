# ***********************************
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from skimage import filters
import pickle
import time
import datetime
import cv2 as cv
# ***********************************
DataRoot = 'data/AnnotationDS2022-08-04_17_32/'
DataBar = 'T00000B000.npy'


def generateDSet(data_root=DataRoot, data_bar=DataBar):
    seed = np.random.randint(3000)
    test_data001 = np.load(data_root + '001/' + data_bar)
    test_data002 = np.load(data_root + '002/' + data_bar)
    test_data = np.concatenate((test_data001, test_data002))
    test_label = np.concatenate((np.ones((test_data001.shape[0])), 2 * np.ones((test_data002.shape[0])))).astype('int')
    np.random.seed(seed)
    np.random.shuffle(test_data)
    np.random.seed(seed)
    np.random.shuffle(test_label)

    return test_data, test_label


x, y = generateDSet()

'''
clf = svm.SVC()
clf.fit(x, y)

clf2 = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=0, n_jobs=4)
clf2.fit(x, y)

x2, y2 = generateDSet(data_bar='T00000B001.npy')
y2H = clf.predict(x2)
y2H2 = clf2.predict(x2)

testImg = np.load('data/AnnotationDS2022-08-03_20_11/Processed_Stack_T00000.npy')
testX = testImg[:, :, :, 0].reshape((48, 2160 * 2560))
testX = np.moveaxis(testX, 1, 0)

testY = clf.predict(testX)
testYNDA = testY.reshape((2160, 2560))
plt.imshow(testYNDA, cmap='gray')
testY2 = clf2.predict(testX)
testYNDA2 = testY2.reshape((2160, 2560))
plt.imshow(testYNDA2, cmap='gray')
'''

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
print('Time cosumed: {} secs'.format(time.time()-t1))
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

