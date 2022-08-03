# ***********************************
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
# ***********************************
DataRoot = 'data/AnnotationDS2022-08-03_20_11/'
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
