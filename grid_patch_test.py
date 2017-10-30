from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, maximum, Maximum
from keras.layers import Conv2D, MaxPooling2D, Reshape, Lambda
from keras import backend as K
from scipy import misc
from numpy import array
import numpy as np
import tools

batch_size = 16
num_classes = 32
epochs = 474
patch_size = 32

# input image dimensions
img_rows, img_cols = 64, 64

# the data, shuffled and split between train and test sets
#read in training data
class_num = 32
train_ele_num = 52
x_train = []
y_train = []
for i in range(class_num):
	for j in range(train_ele_num):
		filename = 'selective_32/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg'
		img = misc.imread(filename)
		img_flattened = np.reshape(img, img.shape[0]*img.shape[1])
		x_train.append(img_flattened)
		y_train.append(i)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

#read in testing data
ele_num = 64
x_test = []
y_test = []
for i in range(class_num):
	for j in range(train_ele_num, ele_num):
		filename = 'selective_32/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg'
		img = misc.imread(filename)
		img_flattened = np.reshape(img, img.shape[0]*img.shape[1])
		x_test.append(img_flattened)
		y_test.append(i)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, patch_size, patch_size)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (patch_size, patch_size, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input = Input(shape=input_shape)

#hidden layer 0
left = Conv2D(filters=96, kernel_size=(8, 8), activation='relu', padding='same')(input)
right = Conv2D(filters=96, kernel_size=(8, 8), activation='relu', padding='same')(input)
h0 = maximum([left, right])
#apply max pooling
h0 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h0)
#dropout on h0
h0 = Dropout(0.2, name='dropout_h0')(h0)

#hidden layer 1
left = Conv2D(filters=192, kernel_size=(8, 8), activation='relu', padding='same')(h0)
right = Conv2D(filters=192, kernel_size=(8, 8), activation='relu', padding='same')(h0)
h1 = maximum([left, right])
#apply max pooling
h1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(h1)


#hidden layer 2
left = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')(h1)
right = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')(h1)
h2 = maximum([left, right])
#apply max pooling
h2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h2)


#maxout layer
#x = Lambda(lambda x: K.max(h2, 3, True))(h2)
x_shape = h2.get_shape().as_list()
x = Reshape((x_shape[1]*x_shape[2]*x_shape[3],))(h2)
print(x.shape)
m1 = Dense(500)(x)
m2 = Dense(500)(x)
m3 = Dense(500)(x)
m4 = Dense(500)(x)
m5 = Dense(500)(x)
maxout = maximum([m1, m2, m3, m4, m5])
print(maxout.shape)

x = Dense(num_classes, activation='softmax')(maxout)
print(x.shape)

final = Model(input, x)
print(final.summary())
final.load_weights('brodatz_model.h5', by_name=True)

final.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

x_test_grids = [tools.extract_grid_patches(im, patch_size) for im in x_test]
final_prediction = []
for grids in x_test_grids:
	predictions = [np.argmax(p) for p in final.predict(grids)]
	final_prediction.append(max(set(predictions), key=predictions.count))
y_test = [np.argmax(y) for y in y_test]
y_test = np.ravel(y_test)
final_prediction = np.ravel(final_prediction)
print('accuracy:'+str(sum(y_test == final_prediction)/y_test.shape[0]))