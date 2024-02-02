# data gathering
import cv2
import numpy as np
from keras.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, f1_score
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape
y_train.shape

plt.imshow(x_train[23212], plt.get_cmap('binary'))

# visualize the img
x = 1
for i in range(30):
    plt.subplot(6, 6, x)
    plt.imshow(x_train[i], plt.get_cmap('binary'))
    x += 1


# preprocessing the data
y_cat_tr = to_categorical(y_train)
y_cat_ts = to_categorical(y_test)

x_train = x_train/255
x_test = x_test/255

# model building

sn = Sequential()

sn.add(Convolution2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1)))
sn.add(MaxPooling2D(pool_size=(2, 2)))
sn.add(BatchNormalization())
sn.add(Dropout(0.2))

sn.add(Convolution2D(filters=16, kernel_size=(3, 3)))
sn.add(MaxPooling2D(pool_size=(2, 2)))
sn.add(BatchNormalization())
sn.add(Dropout(0.2))


sn.add(Convolution2D(filters=16, kernel_size=(3, 3)))
sn.add(MaxPooling2D(pool_size=(2, 2)))
sn.add(BatchNormalization())
sn.add(Dropout(0.2))


sn.add(Flatten())

sn.add(Dense(850, activation='relu'))
sn.add(Dense(800, activation='relu'))
sn.add(Dense(750, activation='relu'))
sn.add(Dense(700, activation='relu'))
sn.add(Dense(650, activation='relu'))

sn.add(Dense(10, activation='softmax'))


# callback
Early = EarlyStopping(monitor='val_loss', patience=7)

# compile
sn.compile(loss='categorical_crossentropy',
           optimizer='adam', metrics=['accuracy'])

# fitting the model
hist = sn.fit(x_train, y_cat_tr, validation_split=0.2, epochs=15)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])


# evalution of data

y_pred = sn.predict(x_test)
y_pred

y_pred1 = np.argmax(y_pred, axis=-1)
y_pred1

acc = accuracy_score(y_pred1, y_test)
clf = classification_report(y_pred1, y_test)
cnf = multilabel_confusion_matrix(y_pred1, y_test)

print('Accuracy:', acc)
print('classification_report:\n', clf)
print('Confusion_matrix:\n', cnf)


# unseen data
img = cv2.imread('/content/Example_7.jpg', cv2.IMREAD_GRAYSCALE)
img

plt.imshow(img, plt.get_cmap('binary'))
img = ~img
plt.imshow(img, plt.get_cmap('binary'))
img1 = img/255  # scaling
img2 = np.array([img1])
result = sn.predict(img2)

print(result)
