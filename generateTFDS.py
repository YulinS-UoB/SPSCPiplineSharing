# ***********************************
import tensorflow as tf
import numpy as np
# ***********************************


def fetchData(data_prefix, time=0, bar=0, seed=1428, batchsize=128):
    data_l001 = np.load('data/{}/dataset/001/T{:05d}B{:03d}.npy'.format(data_prefix, time, bar))
    data_l002 = np.load('data/{}/dataset/002/T{:05d}B{:03d}.npy'.format(data_prefix, time, bar))
    datal_mix_x = np.concatenate((data_l001[0:9984, :], data_l002[0:9984, :]), axis=0)
    datal_mix_y = np.concatenate((np.zeros((data_l001.shape[0], )), np.ones((data_l002.shape[0], ))), axis=0)
    np.random.seed(seed)
    np.random.shuffle(datal_mix_x)
    np.random.seed(seed)
    np.random.shuffle(datal_mix_y)
    feature_ds = tf.data.Dataset.from_tensor_slices(datal_mix_x)
    label_ds = tf.data.Dataset.from_tensor_slices(datal_mix_y)
    image_label_ds = tf.data.Dataset.zip((feature_ds, label_ds))
    ds = image_label_ds.batch(batchsize)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
