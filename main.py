import numpy as np
from feature import prepare_feature_matrix, check_outliers, extract_features_and_label
from resource import Observatory
import matplotlib.pyplot as plt
from math import floor
import tensorflow as tf
from keras import Sequential, layers


station = Observatory('BOU')
station.parse()

# Feature Extraction
check_outliers(station)
X, y = prepare_feature_matrix(station)

plt.title('Feature Matrix for Model Training')
plt.pcolormesh(X)
plt.show()

plt.hist(y)
plt.show()

print(X.shape, y.shape)
N = len(X)
n_train = floor(N*0.8)
n_val = floor(N*0.1)
X_train = np.array(X[:n_train])
X_val = np.array(X[n_train:n_train+n_val])
X_test = np.array(X[n_train+n_val:])
y_train = np.array(y[:n_train])
y_val = np.array(y[n_train:n_train+n_val])
y_test = np.array(y[n_train+n_val:])

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))

BATCH_SIZE = 120
input_shape = ()
for f, l in train_set.take(1):
    input_shape = f.shape
print(input_shape)
train_set = train_set.shuffle(100).batch(BATCH_SIZE)
val_set = val_set.batch(BATCH_SIZE)

model = Sequential([
    layers.Input(shape=input_shape),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(36, activation='relu'),
    layers.Dense(1),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=['mean_absolute_error']
)

EPOCHS = 60
history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

y_pred = model.predict(np.array(X_test))
mae = tf.keras.metrics.MeanAbsoluteError()
mae.reset_state()
mae.update_state(y_test, y_pred)
print(mae.result().numpy())

# yp_max = np.zeros(y_pred.shape[0])
# yp_smax = np.zeros(y_pred.shape[0])
# for i, ypi in enumerate(y_pred_ind):
#     yp_max[i] = ypi[-1]
#     yp_smax[i] = ypi[-2]


plt.plot(y_test[:300])
plt.plot(y_pred[:300])
plt.title('Validation of our model')
plt.xlabel('Time Slots (3 Hours Window)')
plt.ylabel('\u0394T (nT)')
plt.legend(['True Value', 'Predicted Value'])
plt.show()

#
# features = []
# recent_day = station.data.shape[0]
# for s in range(0, 8):
#     features.append(ex)