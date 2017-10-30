from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, maximum
from keras.layers import Conv2D, MaxPooling2D, Reshape, Lambda
from keras import backend as K
from scipy import misc
from skimage import color
import numpy as np
import tools

#hyper-parameters
batch_size = 128
num_classes = 5
epochs = 474
patch_size = 32
resize_factor = (64,64)

#load data
dir_path = 'C:\\Users\\RL\\Desktop\\Kitty\\master0\\lab\\read_papers\\implementation\\data'
(x_train, y_train), (x_test, y_test) = tools.read_resized_texture_data(dir_path, resize_factor)

#convert to grayvalue images
x_train = [color.rgb2gray(im) for im in x_train]
x_test = [color.rgb2gray(im) for im in x_test]

#resize images
x_train = [misc.imresize(im, resize_factor) for im in x_train]
x_test = [misc.imresize(im, resize_factor) for im in x_test]

#normalize 
x_train = [im.astype('float32')/255 for im in x_train]
x_test = [im.astype('float32')/255 for im in x_test]

#convert to numpy array
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

if K.image_data_format() == 'channels_first':
    input_shape = (1, patch_size, patch_size)
else:
    input_shape = (patch_size, patch_size, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#define network
input = Input(shape=input_shape)

#hidden layer 0
left = Conv2D(filters=96, kernel_size=(8, 8), activation='relu', padding='same', name='h0_l')(input)
right = Conv2D(filters=96, kernel_size=(8, 8), activation='relu', padding='same', name='h0_r')(input)
h0 = maximum([left, right])
#apply max pooling
h0 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='pool0')(h0)


#hidden layer 1
left = Conv2D(filters=192, kernel_size=(8, 8), activation='relu', padding='same', name='h1_l')(h0)
right = Conv2D(filters=192, kernel_size=(8, 8), activation='relu', padding='same', name='h1_r')(h0)
h1 = maximum([left, right])
#apply max pooling
h1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='pool1')(h1)


#hidden layer 2
left = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same', name='h2_l')(h1)
right = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same', name='h2_r')(h1)
h2 = maximum([left, right])
#apply max pooling
h2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h2)


#maxout layer
#x = Lambda(lambda x: K.max(h2, 3, True))(h2)
x_shape = h2.get_shape().as_list()
x = Reshape((x_shape[1]*x_shape[2]*x_shape[3],))(h2)
print(x.shape)
m1 = Dense(500, name='dense1')(x)
m2 = Dense(500, name='dense2')(x)
m3 = Dense(500, name='dense3')(x)
m4 = Dense(500, name='dense4')(x)
m5 = Dense(500, name='dense5')(x)
maxout = maximum([m1, m2, m3, m4, m5])

x = Dense(num_classes, activation='softmax', name='output')(maxout)
print(x.shape)

final = Model(input, x)
print(final.summary())

#sgd = keras.optimizers.SGD(lr=0.17, decay=1e-6, momentum=0.05, nesterov=True)
final.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

for i in range(epochs):
	print('Epoch '+str(i))
	#extract random patch for each training image
	x_train_patches = [tools.random_image_crop(im, patch_size) for im in x_train]
	x_test_patches = [tools.random_image_crop(im, patch_size) for im in x_test]
	x_train_patches = np.asarray(x_train_patches)
	x_test_patches = np.asarray(x_test_patches)
	
	final.fit(x_train_patches, y_train,
			  batch_size=batch_size,
			  epochs=1,
			  verbose=1,
			  validation_data=(x_test_patches, y_test))

final.save('data_64_64_model.h5')
x_test_patches = np.asarray([tools.random_image_crop(im, patch_size) for im in x_test])
score = final.evaluate(x_test_patches, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])