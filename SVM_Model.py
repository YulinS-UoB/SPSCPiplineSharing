# ***********************************
import tensorflow as tf
import generateTFDS
import numpy as np
from tqdm import tqdm
from skimage import morphology
from scipy import ndimage
# ***********************************

ds_train_1 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=0)
ds_train_2 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=1)
ds_train_3 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=2)
ds_train_4 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=3)
ds_train_5 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=4)
ds_val = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=5)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1392,), batch_size=128),
    tf.keras.layers.experimental.RandomFourierFeatures(
        output_dim=4096,
        kernel_initializer='gaussian'),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),

])
model.compile(
    optimizer='adam',
    loss='hinge',
    metrics=['categorical_accuracy']
)


model.fit(ds_train_1, epochs=20)
model.fit(ds_train_2, epochs=20)
# model.fit(ds_train_3, epochs=20)
# model.fit(ds_train_4, epochs=20)
# model.fit(ds_train_5, epochs=20)
model.evaluate(ds_val)

ProcessedStk = np.load('data/AnnotationDS2022-07-29_23_40/Processed_Stack_T00000.npy')

visionRad = 1
generateMask = [np.empty((0, 0)), np.empty((0, 0))]  # 0: Shape of the mask; 1: Value of the mask
generateMask[0] = morphology.disk(visionRad * 3)
distance_mask = np.ones(generateMask[0].shape, dtype='float32')
distance_mask[visionRad * 3, visionRad * 3] = 0
distance_mask = ndimage.distance_transform_edt(
    np.exp(-np.square(distance_mask) / (2 * np.square(visionRad))) /
    (visionRad * np.sqrt(2 * np.pi))
)
generateMask[1] = generateMask[0] * distance_mask


def generatePrdctImg(res_stack, generate_mask, prdct_model):
    focus_rad = (generate_mask[0].shape[0] - 1) // 2
    prdct_img = np.zeros((res_stack.shape[1], res_stack.shape[2]))
    padded_feature = np.pad(res_stack, ((0, 0), (focus_rad, focus_rad), (focus_rad, focus_rad), (0, 0)),
                            mode='reflect')
    print(padded_feature.shape)
    nd_filter = np.zeros((res_stack.shape[0], generate_mask[0].shape[0], generate_mask[0].shape[1],
                          res_stack.shape[3]))
    noticed_loc = np.where(generate_mask[0] != 0)
    for t in range(res_stack.shape[0]):
        for c in range(res_stack.shape[3]):
            nd_filter[t, :, :, c] = generate_mask[1]
    for x in tqdm(range(res_stack.shape[1])):
        batchid = 0
        batched_feature = np.zeros((128, 1392))
        for y in range(res_stack.shape[2]):
            x_idx = x + focus_rad
            y_idx = y + focus_rad
            feature_region = padded_feature[:, (x_idx - focus_rad):(x_idx + focus_rad + 1),
                                            (y_idx - focus_rad):(y_idx + focus_rad + 1), :]
            noticed_feature = ((feature_region * nd_filter)[:, noticed_loc[0], noticed_loc[1], :]).flatten()
            batched_feature[batchid, :] = noticed_feature
            batchid = batchid + 1
            if batchid == 128:
                prdct_img[x_idx - focus_rad, (y_idx - focus_rad - 127):(y_idx - focus_rad + 1)] = \
                    prdct_model.predict(batched_feature, verbose=0)[:, 0]
                batchid = 0
                batched_feature = np.zeros((128, 1392))
            elif y == res_stack.shape[2] - 1:
                prdct_img[x_idx - focus_rad, (y_idx - focus_rad - batchid + 1):(y_idx - focus_rad + 1)] = \
                    prdct_model.predict(batched_feature, verbose=0)[0:batchid, 0]

    return prdct_img


imgTest = generatePrdctImg(ProcessedStk, generateMask, model)
np.save('imgTest.npy', imgTest)
