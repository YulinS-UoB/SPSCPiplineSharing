# ***********************************
import tensorflow as tf
import pathlib
import random
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE
# ***********************************

data_root = pathlib.Path('data/AnnotationDS2022-07-21_23_14/dataset')


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

img_path = all_image_paths[0]
img_raw = np.load(img_path)
img_tensor = tf.convert_to_tensor(img_raw, dtype=tf.float32)
print(img_tensor.shape)
print(img_tensor.dtype)


def preprocess_image(image_path):
    image = np.load(image_path)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

image_ds = path_ds.map(tf.autograph.experimental.do_not_convert(
        lambda item: tf.numpy_function(preprocess_image, [item], tf.float32)))

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = image_label_ds.shuffle(buffer_size=5000)
ds = ds.repeat()
ds = ds.batch(32)
ds = ds.prefetch(buffer_size=AUTOTUNE)
