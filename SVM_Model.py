# ***********************************
import tensorflow as tf
import generateTFDS
# ***********************************

ds_train = generateTFDS.ds.skip(5000)
ds_val = generateTFDS.ds.take(5000)

batch_size = 32

model = tf.keras.Sequential([
  tf.keras.Input(shape=(1392,), batch_size=32),
  tf.keras.layers.RandomFourierFeatures(
      output_dim=4096,
      scale=10.,
      kernel_initializer='gaussian'),
  tf.keras.layers.Dense(units=10),
])
model.compile(
    optimizer='adam',
    loss='hinge',
    metrics=['categorical_accuracy']
)

for x_train, y_train in ds_train:
    model.fit(x_train, y_train, epochs=5)

for x_val, y_val in ds_val:
    model.fit(x_val, y_val, epochs=5)
