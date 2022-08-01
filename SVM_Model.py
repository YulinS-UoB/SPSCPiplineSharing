# ***********************************
import tensorflow as tf
import generateTFDS
# ***********************************

ds_train_1 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=0)
ds_train_2 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=1)
ds_train_3 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=2)
ds_train_4 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=3)
ds_train_5 = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=4)
ds_val = generateTFDS.fetchData(data_prefix='AnnotationDS2022-07-29_23_40', time=0, bar=5)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1392,), batch_size=32),
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
model.fit(ds_train_3, epochs=20)
model.fit(ds_train_4, epochs=20)
model.fit(ds_train_5, epochs=20)
model.evaluate(ds_val)
