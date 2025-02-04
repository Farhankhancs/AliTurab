import numpy as np
from numpy.random import seed
seed(1)

# Import TensorFlow and Keras utilities
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(2)
import os
import seaborn as sn
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Lambda
from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import Normalizer
from tensorflow.keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
import h5py
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib as mpl
import matplotlib.pylab as plt
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import plot_model
import time

classifications = 2
epochs = 50
dimension = dimentions

# Load dataset
dataset1 = np.loadtxt('dataset.csv', delimiter=",")
ticklabels = ['label1', 'label2']

# Data Split
X = dataset1[:, :]
Y = dataset1[:, ]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.80, random_state=0)

# Convert labels to categorical format
Y_train = to_categorical(y_train-1, classifications)
Y_test = to_categorical(y_test-1, classifications)

# Reshape the input data for CNN
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Model definition
model = Sequential()
model.add(Convolution1D(32, 4, padding="same", activation="relu", input_shape=(dimension, 1)))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Dropout(0.6))
model.add(Convolution1D(64, 4, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Dropout(0.7))
model.add(LSTM(32))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.7))
model.add(Dense(classifications, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()


# Model training
history = model.fit(X_train, Y_train, epochs=epochs, verbose=1, batch_size=32, validation_data=(X_test, Y_test), callbacks=[checkpointer, csv_logger, time_callback])

# Time summary
print(time_callback.times)
total = sum(time_callback.times)
print("Total Time of Training=")
print("%.3f" % total)

# Save model
model.save("cnn.hdf5")
model.summary()

# Plotting training history
fig, ax = plt.subplots()
plt.plot(history.history['accuracy'], 'b-')
plt.plot(history.history['val_accuracy'], 'r--')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
ax.set_xticks(range(0, epochs+4, 50))
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('model_accuracy.png', dpi=300)

fig, ax = plt.subplots()
plt.plot(history.history['loss'], 'y-')
plt.plot(history.history['val_loss'], 'g--')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax.set_xticks(range(0, epochs+4, 50))
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('model_loss.png', dpi=300)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

# Classification report
predi = model.predict(X_test)
pred = np.argmax(predi, axis=1)
y_test2 = np.argmax(Y_test, axis=1)
print('Report:')
print(classification_report(y_test2, pred))

# Confusion Matrix
cm = confusion_matrix(y_test2, pred)
cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
print('Confusion Matrix with Normalization:')
print(cm_normalized)

fig, ax = plt.subplots(figsize=(10, 7))
sn.heatmap(cm_normalized, annot=True, annot_kws={"size": 14, "fontweight": 'bold'}, cmap='Blues', cbar=False)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
ax.set_xticks(np.arange(len(ticklabels)) + 0.3)
ax.set_yticks(np.arange(len(ticklabels)) + 0.3)
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
fig.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
