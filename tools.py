import random
from scipy import misc
import numpy as np
from os import listdir
import imghdr

def random_image_crop(im, patch_size):
	if(len(im.shape) == 2):	#no specified channel
		row, col = im.shape
		channel = 1
	else:
		row, col, channel = im.shape
	random.seed()
	start_row = random.randint(0, row - patch_size)
	start_col = random.randint(0, col - patch_size)
	result = im[start_row:start_row+patch_size, start_col:start_col+patch_size]
	result = result.reshape(result.shape[0], result.shape[1], channel)
	return result
	
def extract_grid_patches(im, patch_size):
	if(len(im.shape) == 2):	#no specified channel
		row, col = im.shape
		channel = 1
	else:
		row, col, channel = im.shape
	start_rows = row/patch_size
	start_cols = col/patch_size
	results = []
	for i in range(int(start_rows)):
		for j in range(int(start_cols)):
			results.append(im[i:i+patch_size, j:j+patch_size])
		if(col%patch_size > 0):
			results.append(im[i:i+patch_size, col-patch_size:col])
	if(row%patch_size > 0):
		for j in range(int(start_cols)):
			results.append(im[row-patch_size:row, j:j+patch_size])
		if(col%patch_size > 0):
			results.append(im[row-patch_size:row, col-patch_size:col])
	patches = [im.reshape(im.shape[0], im.shape[1], channel) for im in results]
	patches = np.asarray(results)
	return patches
	
def read_texture_data(dir_path):
	x_train = []
	y_train = []
	x_test= []
	y_test = []
	images = []
	labels = []

	#get all label names
	dirs = listdir(dir_path)
	class_index = 0

	#read in all images
	for d in dirs:
		image_path = dir_path + '\\' + d
		print(d)
		files = [image_path+'\\'+f for f in listdir(image_path) if (imghdr.what(image_path+'\\'+f) != None)]
		images = images + [misc.imread(im) for im in files]
		labels = labels + [class_index]*len(files)
		class_index = class_index + 1
	
	#shuffle the data
	combined = list(zip(images, labels))
	random.shuffle(combined)
	images, labels = zip(*combined)
	
	x_train = images[0:int(len(images)*0.75)]
	x_test = images[int(len(images)*0.75):len(images)]
	y_train = labels[0:int(len(labels)*0.75)]
	y_test = labels[int(len(labels)*0.75):len(labels)]
	
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	
	return (x_train, y_train), (x_test, y_test)

def read_resized_texture_data(dir_path, resize_factor):
	x_train = []
	y_train = []
	x_test= []
	y_test = []
	images = []
	labels = []

	#get all label names
	dirs = listdir(dir_path)
	class_index = 0

	#read in all images
	for d in dirs:
		image_path = dir_path + '\\' + d
		print(d)
		files = [image_path+'\\'+f for f in listdir(image_path) if (imghdr.what(image_path+'\\'+f) != None)]
		images = images + [misc.imresize(misc.imread(im), resize_factor) for im in files]
		labels = labels + [class_index]*len(files)
		class_index = class_index + 1
	
	#shuffle the data
	combined = list(zip(images, labels))
	random.shuffle(combined)
	images, labels = zip(*combined)
	
	x_train = images[0:int(len(images)*0.75)]
	x_test = images[int(len(images)*0.75):len(images)]
	y_train = labels[0:int(len(labels)*0.75)]
	y_test = labels[int(len(labels)*0.75):len(labels)]
	
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	
	return (x_train, y_train), (x_test, y_test)